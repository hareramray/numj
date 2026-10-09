!> numj native kernels: float64, contiguous, C-interoperable (ISO_C_BINDING / bind(C)).
!>
!> Contract (enforced by the Java layer, NOT re-checked here):
!>   * all array pointers are valid for the stated extents and 8-byte aligned;
!>   * 2-D arrays are row-major (C order) with contiguous rows: a [nrows x ncols] array whose rows are
!>     ld >= ncols elements apart is seen here as x(ld, nrows), so row i is the contiguous x(1:ncols, i);
!>   * arrays written by a routine do not overlap any other argument of that routine
!>     (in-place row normalization has its own single-argument entry point);
!>   * read-only arguments may alias each other freely.
!> No routine keeps a pointer after it returns, allocates persistent state, or calls back into Java.
!>
!> Summation order (deterministic, independent of thread count):
!>   the input is cut into blocks of BLOCK elements; inside a block, element j is added to
!>   partial sum lane mod(j-1, LANES)+1; the LANES partials are combined by a fixed pairwise tree;
!>   block results are combined by a fixed pairwise tree. Any rounding path therefore has at most
!>   ceil(BLOCK/LANES) + log2(LANES) + ceil(log2(nblocks)) additions.
module numj_kernels
  use, intrinsic :: iso_c_binding, only: c_double, c_int, c_int64_t, c_char, c_null_char
  use, intrinsic :: iso_fortran_env, only: compiler_version, compiler_options
  use, intrinsic :: ieee_arithmetic, only: ieee_is_nan
  implicit none
  private

  integer, parameter :: dp = c_double
  integer, parameter :: LANES = 16
  integer(c_int64_t), parameter :: BLOCK = 4096_c_int64_t
  !> Stack scratch for block partials up to this many blocks (n <= 1 Mi elements); heap above.
  integer(c_int64_t), parameter :: STACK_BLOCKS = 256_c_int64_t
  !> Row sums of squares below this (or +Inf) are recomputed with scaling (see row_norm).
  real(dp), parameter :: SS_LO = 2.0_dp**(-968)
  integer, parameter :: ABI_VERSION = 2
  integer, parameter :: K_SQDIST = 1, K_MULADD = 2, K_SUMSQ = 3, K_SUM = 4   ! block-kernel selectors

  public :: numj_abi_version, numj_build_info, numj_sqdist, numj_sumsq_muladd, numj_sum, numj_sqdist_rows, &
            numj_normalize_rows, numj_normalize_rows_inplace
  ! Building blocks shared with numj_reduce (same summation order everywhere).
  public :: dp, LANES, BLOCK, STACK_BLOCKS, K_SQDIST, K_MULADD, K_SUM, reduce_blocks, lane_tree, pairwise

contains

  ! ----------------------------------------------------------------------------------------------
  ! Block kernels (one block of m <= BLOCK elements). Lanes are written as fixed-length array
  ! operations so the compiler can map them onto SIMD registers without reassociating anything.
  ! ----------------------------------------------------------------------------------------------

  pure function lane_tree(acc) result(s)
    real(dp), intent(in) :: acc(LANES)
    real(dp) :: s, t(LANES/2)
    t(1:8) = acc(1:8) + acc(9:16)
    t(1:4) = t(1:4) + t(5:8)
    t(1:2) = t(1:2) + t(3:4)
    s = t(1) + t(2)
  end function lane_tree

  pure function blk_sqdist(a, b, m) result(s)
    integer(c_int64_t), intent(in) :: m
    real(dp), intent(in) :: a(m), b(m)
    real(dp) :: s, acc(LANES), d(LANES)
    integer(c_int64_t) :: i, m0
    acc = 0.0_dp
    m0 = m - mod(m, int(LANES, c_int64_t))
    do i = 1, m0, LANES
      d = a(i:i+LANES-1) - b(i:i+LANES-1)
      acc = acc + d*d
    end do
    if (m > m0) then                ! tail -> lanes 1..m-m0; padding adds +0 (bitwise no-op)
      d = 0.0_dp
      d(1:m-m0) = a(m0+1:m) - b(m0+1:m)
      acc = acc + d*d
    end if
    s = lane_tree(acc)
  end function blk_sqdist

  pure function blk_sumsq_muladd(a, b, c, m) result(s)
    integer(c_int64_t), intent(in) :: m
    real(dp), intent(in) :: a(m), b(m), c(m)
    real(dp) :: s, acc(LANES), t(LANES)
    integer(c_int64_t) :: i, m0
    acc = 0.0_dp
    m0 = m - mod(m, int(LANES, c_int64_t))
    do i = 1, m0, LANES
      t = a(i:i+LANES-1)*b(i:i+LANES-1) + c(i:i+LANES-1)
      acc = acc + t*t
    end do
    if (m > m0) then
      t = 0.0_dp
      t(1:m-m0) = a(m0+1:m)*b(m0+1:m) + c(m0+1:m)
      acc = acc + t*t
    end if
    s = lane_tree(acc)
  end function blk_sumsq_muladd

  pure function blk_sumsq(x, m) result(s)
    integer(c_int64_t), intent(in) :: m
    real(dp), intent(in) :: x(m)
    real(dp) :: s, acc(LANES), v(LANES)
    integer(c_int64_t) :: i, m0
    acc = 0.0_dp
    m0 = m - mod(m, int(LANES, c_int64_t))
    do i = 1, m0, LANES
      acc = acc + x(i:i+LANES-1)*x(i:i+LANES-1)
    end do
    if (m > m0) then
      v = 0.0_dp
      v(1:m-m0) = x(m0+1:m)
      acc = acc + v*v
    end if
    s = lane_tree(acc)
  end function blk_sumsq

  !> Plain sum. Lanes start at +0.0 (NumPy's identity: a sum of only -0.0 values is +0.0); the tail is
  !> padded with +0.0, which is a bitwise no-op because a lane that started at +0.0 can never hold -0.0.
  pure function blk_sum(x, m) result(s)
    integer(c_int64_t), intent(in) :: m
    real(dp), intent(in) :: x(m)
    real(dp) :: s, acc(LANES), v(LANES)
    integer(c_int64_t) :: i, m0
    acc = 0.0_dp
    m0 = m - mod(m, int(LANES, c_int64_t))
    do i = 1, m0, LANES
      acc = acc + x(i:i+LANES-1)
    end do
    if (m > m0) then
      v = 0.0_dp
      v(1:m-m0) = x(m0+1:m)
      acc = acc + v
    end if
    s = lane_tree(acc)
  end function blk_sum

  !> In-place pairwise reduction with a fixed shape: p(1)+p(2), p(3)+p(4), ... repeated.
  function pairwise(p, n) result(s)
    integer(c_int64_t), intent(in) :: n
    real(dp), intent(inout) :: p(n)
    real(dp) :: s
    integer(c_int64_t) :: len, half, i
    len = n
    do while (len > 1)
      half = len / 2
      do i = 1, half
        p(i) = p(2*i-1) + p(2*i)
      end do
      if (mod(len, 2_c_int64_t) == 1) then
        p(half+1) = p(len)
        len = half + 1
      else
        len = half
      end if
    end do
    s = p(1)
  end function pairwise

  ! ----------------------------------------------------------------------------------------------
  ! Whole-vector reductions: blocks -> partials -> pairwise. op selects the block kernel.
  ! ----------------------------------------------------------------------------------------------


  !> Single-block inputs (the common case for rows and small vectors) take the direct path; the
  !> multi-block machinery (scratch arrays, optional OpenMP) lives in reduce_multi.
  function reduce_blocks(op, n, a, b, c, nthreads) result(s)
    integer, intent(in) :: op
    integer(c_int64_t), intent(in) :: n
    real(dp), intent(in) :: a(*), b(*), c(*)
    integer(c_int), intent(in) :: nthreads
    real(dp) :: s
    if (n <= BLOCK) then
      s = blk(op, 1_c_int64_t, max(n, 0_c_int64_t), a, b, c)   ! m = 0 gives +0.0
    else
      s = reduce_multi(op, n, a, b, c, nthreads)
    end if
  end function reduce_blocks

  function reduce_multi(op, n, a, b, c, nthreads) result(s)
    integer, intent(in) :: op
    integer(c_int64_t), intent(in) :: n
    real(dp), intent(in) :: a(*), b(*), c(*)
    integer(c_int), intent(in) :: nthreads
    real(dp) :: s
    real(dp) :: stackp(STACK_BLOCKS)
    real(dp), allocatable :: heapp(:)
    integer(c_int64_t) :: nb

    nb = (n + BLOCK - 1) / BLOCK
    if (nb <= STACK_BLOCKS) then
      call fill(stackp)
      s = pairwise(stackp, nb)
    else
      allocate(heapp(nb))
      call fill(heapp)
      s = pairwise(heapp, nb)
    end if

  contains

    subroutine fill(p)
      real(dp), intent(out) :: p(*)
      integer(c_int64_t) :: k, lo
      if (nthreads > 1) then
        !$omp parallel do num_threads(nthreads) schedule(dynamic, 4) private(lo)
        do k = 1, nb
          lo = (k-1)*BLOCK + 1
          p(k) = blk(op, lo, min(BLOCK, n - lo + 1), a, b, c)
        end do
        !$omp end parallel do
      else
        do k = 1, nb
          lo = (k-1)*BLOCK + 1
          p(k) = blk(op, lo, min(BLOCK, n - lo + 1), a, b, c)
        end do
      end if
    end subroutine fill

  end function reduce_multi

  function blk(op, lo, m, a, b, c) result(s)
    integer, intent(in) :: op
    integer(c_int64_t), intent(in) :: lo, m
    real(dp), intent(in) :: a(*), b(*), c(*)
    real(dp) :: s
    select case (op)
    case (K_SQDIST)
      s = blk_sqdist(a(lo), b(lo), m)         ! sequence association: no copies
    case (K_MULADD)
      s = blk_sumsq_muladd(a(lo), b(lo), c(lo), m)
    case (K_SUM)
      s = blk_sum(a(lo), m)
    case default
      s = blk_sumsq(a(lo), m)
    end select
  end function blk

  ! ----------------------------------------------------------------------------------------------
  ! Row norms with overflow/underflow-safe fallback.
  !   mode 0: zero norm or empty row  -> output row is a copy of the input (all +/-0)
  !   mode 1: plain                   -> y = x / nrm
  !   mode 2: rescaled                -> y = (x / amax) / rs, nrm = amax * rs (may be +Inf only if
  !                                      the true norm exceeds huge())
  ! NaN anywhere in the row -> nrm = NaN, mode 1 (y all NaN).
  ! +/-Inf in the row (no NaN) -> nrm = +Inf, mode 1 (finite -> +/-0, +/-Inf -> NaN; IEEE x/Inf).
  ! ----------------------------------------------------------------------------------------------

  subroutine row_norm(x, m, nrm, amax, rs, mode)
    integer(c_int64_t), intent(in) :: m
    real(dp), intent(in) :: x(m)
    real(dp), intent(out) :: nrm, amax, rs
    integer, intent(out) :: mode
    real(dp) :: ss
    integer(c_int64_t) :: j

    amax = 0.0_dp
    rs = 1.0_dp
    ss = reduce_blocks(K_SUMSQ, m, x, x, x, 1_c_int)
    if (ieee_is_nan(ss)) then               ! NaN: propagate
      nrm = ss
      mode = 1
      return
    end if
    if (ss >= SS_LO .and. ss <= huge(ss)) then
      nrm = sqrt(ss)
      mode = 1
      return
    end if
    ! Rare path: sum of squares overflowed, underflowed, or the row is zero.
    do j = 1, m
      amax = max(amax, abs(x(j)))
    end do
    if (amax == 0.0_dp) then
      nrm = 0.0_dp
      mode = 0
    else if (amax > huge(amax)) then        ! contains +/-Inf
      nrm = amax
      mode = 1
    else
      ss = 0.0_dp
      do j = 1, m
        ss = ss + (x(j)/amax)**2
      end do
      rs = sqrt(ss)
      nrm = amax * rs
      mode = 2
    end if
  end subroutine row_norm

  ! ----------------------------------------------------------------------------------------------
  ! C ABI
  ! ----------------------------------------------------------------------------------------------

  function numj_abi_version() bind(C, name='numj_abi_version') result(v)
    integer(c_int) :: v
    v = ABI_VERSION
  end function numj_abi_version

  !> Writes "<compiler version> | <compiler options>" (truncated to cap-1 chars, NUL-terminated).
  !> Returns the number of characters written, excluding the NUL.
  function numj_build_info(buf, cap) bind(C, name='numj_build_info') result(len)
    integer(c_int64_t), value :: cap
    character(kind=c_char), intent(out) :: buf(cap)
    integer(c_int64_t) :: len, i
    character(len=:), allocatable :: s
    s = compiler_version() // ' | ' // compiler_options()
    len = min(int(len_trim(s), c_int64_t), cap - 1)
    do i = 1, len
      buf(i) = s(i:i)
    end do
    if (cap > 0) buf(len+1) = c_null_char
  end function numj_build_info

  !> sum_i (a_i - b_i)**2
  function numj_sqdist(a, b, n, nthreads) bind(C, name='numj_sqdist') result(s)
    integer(c_int64_t), value :: n
    integer(c_int), value :: nthreads
    real(c_double), intent(in) :: a(*), b(*)
    real(c_double) :: s
    s = reduce_blocks(K_SQDIST, n, a, b, a, nthreads)
  end function numj_sqdist

  !> sum_i (a_i*b_i + c_i)**2, with a_i*b_i and +c_i rounded separately (no FMA contraction).
  function numj_sumsq_muladd(a, b, c, n, nthreads) bind(C, name='numj_sumsq_muladd') result(s)
    integer(c_int64_t), value :: n
    integer(c_int), value :: nthreads
    real(c_double), intent(in) :: a(*), b(*), c(*)
    real(c_double) :: s
    s = reduce_blocks(K_MULADD, n, a, b, c, nthreads)
  end function numj_sumsq_muladd

  !> sum_i x_i (plain sum, same blocked order as the other reductions)
  function numj_sum(x, n, nthreads) bind(C, name='numj_sum') result(s)
    integer(c_int64_t), value :: n
    integer(c_int), value :: nthreads
    real(c_double), intent(in) :: x(*)
    real(c_double) :: s
    s = reduce_blocks(K_SUM, n, x, x, x, nthreads)
  end function numj_sum

  !> out(i) = sum_j (x(j,i) - q(j))**2 for each row i. Each out(i) is bit-identical to
  !> numj_sqdist(row_i, q, ncols, 1). Rows are ldx >= ncols elements apart.
  subroutine numj_sqdist_rows(q, x, nrows, ncols, ldx, out, nthreads) bind(C, name='numj_sqdist_rows')
    integer(c_int64_t), value :: nrows, ncols, ldx
    integer(c_int), value :: nthreads
    real(c_double), intent(in) :: q(ncols), x(ldx, nrows)
    real(c_double), intent(out) :: out(nrows)
    integer(c_int64_t) :: i
    if (nthreads > 1) then
      !$omp parallel do num_threads(nthreads) schedule(dynamic, 64)
      do i = 1, nrows
        out(i) = reduce_blocks(K_SQDIST, ncols, x(1, i), q, q, 1_c_int)
      end do
      !$omp end parallel do
    else
      do i = 1, nrows
        out(i) = reduce_blocks(K_SQDIST, ncols, x(1, i), q, q, 1_c_int)
      end do
    end if
  end subroutine numj_sqdist_rows

  !> One row, out of place. Explicit arguments (no host association) so the compiler sees
  !> two distinct, contiguous, non-aliasing arrays.
  subroutine normalize_one(x, y, m, nrm)
    integer(c_int64_t), intent(in) :: m
    real(dp), intent(in) :: x(m)
    real(dp), intent(out) :: y(m)
    real(dp), intent(out) :: nrm
    real(dp) :: amax, rs
    integer :: mode
    call row_norm(x, m, nrm, amax, rs, mode)
    select case (mode)
    case (0)
      y = x
    case (1)
      y = x / nrm
    case default
      y = (x / amax) / rs
    end select
  end subroutine normalize_one

  !> One row, in place.
  subroutine normalize_one_inplace(x, m, nrm)
    integer(c_int64_t), intent(in) :: m
    real(dp), intent(inout) :: x(m)
    real(dp), intent(out) :: nrm
    real(dp) :: amax, rs
    integer :: mode
    call row_norm(x, m, nrm, amax, rs, mode)
    select case (mode)
    case (1)
      x = x / nrm
    case (2)
      x = (x / amax) / rs
    end select
  end subroutine normalize_one_inplace

  !> y(:,i) = x(:,i) / ||x(:,i)||_2 per row; see row_norm for zero/NaN/Inf/overflow rules.
  !> norms is optional (pass NULL to skip); if present norms(i) = ||x(:,i)||_2.
  !> Rows of x are ldx >= ncols elements apart, rows of y ldy >= ncols; elements between rows are untouched.
  subroutine numj_normalize_rows(x, ldx, y, ldy, nrows, ncols, norms, nthreads) &
      bind(C, name='numj_normalize_rows')
    integer(c_int64_t), value :: ldx, ldy, nrows, ncols
    integer(c_int), value :: nthreads
    real(c_double), intent(in) :: x(ldx, nrows)
    real(c_double), intent(inout) :: y(ldy, nrows)
    real(c_double), intent(out), optional :: norms(nrows)
    integer(c_int64_t) :: i
    real(dp) :: nrm
    if (nthreads > 1) then
      !$omp parallel do num_threads(nthreads) schedule(dynamic, 64) private(nrm)
      do i = 1, nrows
        call normalize_one(x(1, i), y(1, i), ncols, nrm)
        if (present(norms)) norms(i) = nrm
      end do
      !$omp end parallel do
    else
      do i = 1, nrows
        call normalize_one(x(1, i), y(1, i), ncols, nrm)
        if (present(norms)) norms(i) = nrm
      end do
    end if
  end subroutine numj_normalize_rows

  !> In-place variant of numj_normalize_rows (x is both input and output).
  subroutine numj_normalize_rows_inplace(x, ldx, nrows, ncols, norms, nthreads) &
      bind(C, name='numj_normalize_rows_inplace')
    integer(c_int64_t), value :: ldx, nrows, ncols
    integer(c_int), value :: nthreads
    real(c_double), intent(inout) :: x(ldx, nrows)
    real(c_double), intent(out), optional :: norms(nrows)
    integer(c_int64_t) :: i
    real(dp) :: nrm
    if (nthreads > 1) then
      !$omp parallel do num_threads(nthreads) schedule(dynamic, 64) private(nrm)
      do i = 1, nrows
        call normalize_one_inplace(x(1, i), ncols, nrm)
        if (present(norms)) norms(i) = nrm
      end do
      !$omp end parallel do
    else
      do i = 1, nrows
        call normalize_one_inplace(x(1, i), ncols, nrm)
        if (present(norms)) norms(i) = nrm
      end do
    end if
  end subroutine numj_normalize_rows_inplace

end module numj_kernels

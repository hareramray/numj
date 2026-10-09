!> numj reductions over strided n-d arrays (float64).
!>
!> Summation order (the contract): every reduction sums a *logical sequence* - the reduced elements of one
!> output in C order of the reduced axes - with the blocked algorithm of numj_kernels: blocks of BLOCK
!> elements, element j of a block into lane mod(j, LANES), lanes combined by lane_tree, blocks by pairwise.
!> All paths below (contiguous sequences, column tiles, general strided walk) reproduce that order exactly,
!> so results are bitwise identical for every memory layout, every path and every thread count.
!>
!> Layout conventions (validated by the Java layer): element offsets are relative to the pointer passed in
!> and are never negative; dim 1 is the innermost axis; strides are in elements and may be negative or zero.
module numj_reduce
  use, intrinsic :: iso_c_binding, only: c_double, c_int, c_int64_t
  use numj_kernels, only: dp, LANES, BLOCK, STACK_BLOCKS, K_SQDIST, K_MULADD, K_SUM, reduce_blocks, blk, &
                          lane_tree, pairwise
  implicit none
  private
  public :: numj_reduce_seq, numj_sum_nd

  !> Columns per work item in the outer-axis (column) path: at least TILE (enough work per item), at most
  !> WIDE (WIDE x LANES accumulators = 512 KiB, which stays in a 1.25 MiB L2). With one thread a single wide
  !> group covers up to WIDE columns, so every row of x is streamed exactly once, in memory order.
  integer(c_int64_t), parameter :: TILE = 256_c_int64_t, WIDE = 4096_c_int64_t

contains

  !> Offset of C-order flat index r over n dims (dim 1 fastest).
  pure function unravel_off(r, n, shape, stride) result(off)
    integer(c_int64_t), intent(in) :: r
    integer, intent(in) :: n
    integer(c_int64_t), intent(in) :: shape(n), stride(n)
    integer(c_int64_t) :: off, q
    integer :: k
    off = 0
    q = r
    do k = 1, n
      off = off + mod(q, shape(k)) * stride(k)
      q = q / shape(k)
    end do
  end function unravel_off

  !> Blocked reduction of f(element) over a strided sequence; f = x (K_SUM), (a-b)**2 (K_SQDIST) or
  !> (a*b+c)**2 (K_MULADD). Each block of BLOCK consecutive sequence elements is gathered into a small
  !> contiguous buffer and reduced by the same block kernel (blk) that contiguous inputs use; block results
  !> are combined by pairwise. The result is therefore bit-identical to reduce_blocks on the same values
  !> stored contiguously, while the arithmetic runs in the vectorised kernel instead of a scalar lane loop.
  function seq_gather(op, nd, shape, a, oa, sa, b, ob, sb, c, oc, sc) result(s)
    integer, intent(in) :: op, nd
    integer(c_int64_t), intent(in) :: shape(nd), sa(nd), sb(nd), sc(nd), oa, ob, oc
    real(dp), intent(in) :: a(0:*), b(0:*), c(0:*)
    real(dp) :: s
    ! SMALL: gather buffers up to this size live on the stack. TILED_MIN_N: smallest sequence for the tiled
    ! walk. Paths chosen from measurements (results/nd/RESULTS.md, round 2): tiled for large transposed
    ! inputs, untiled gather for smaller plain sums (smaller fused kernels go to seq_scalar instead).
    integer(c_int64_t), parameter :: SMALL = 1024_c_int64_t, TILE_ELEMS = 32768_c_int64_t, &
                                     TILED_MIN_N = 1048576_c_int64_t
    real(dp) :: stackp(STACK_BLOCKS), ta(SMALL), tb(SMALL), tc(SMALL)
    real(dp), allocatable :: heapp(:), ha(:), hb(:), hc(:)
    integer(c_int64_t) :: n, nbuf, nb, tt
    logical :: onheap, tiled

    n = product(shape)
    if (n == 0) then
      s = 0.0_dp
      return
    end if
    onheap = (n + BLOCK - 1) / BLOCK > STACK_BLOCKS
    if (onheap) allocate(heapp((n + BLOCK - 1) / BLOCK))
    nbuf = min(n, BLOCK)
    nb = 0
    ! Tiled walk: when the logical inner axis has a large stride but axis 2 has unit stride in every operand
    ! (e.g. a transposed C array), read TT neighbouring axis-2 positions per memory row (one cache line, one
    ! page) into a small tile, then feed the TT logical rows in order. Same sequence, same bits.
    tt = 1
    if (nd >= 2) then
      tiled = abs(sa(1)) > 8 .and. sa(2) == 1 .and. (op == K_SUM .or. sb(2) == 1) .and. &
              (op /= K_MULADD .or. sc(2) == 1) .and. shape(2) > 1
      if (tiled) tt = min(8_c_int64_t, shape(2), TILE_ELEMS / shape(1))
    end if
    if (tt >= 2 .and. n >= TILED_MIN_N) then
      allocate(ha(nbuf))
      allocate(hb(merge(nbuf, 1_c_int64_t, op /= K_SUM)))
      allocate(hc(merge(nbuf, 1_c_int64_t, op == K_MULADD)))
      call walk_tiled(ha, hb, hc)
    else if (nbuf <= SMALL) then
      call walk(ta, tb, tc)
    else
      allocate(ha(nbuf))
      allocate(hb(merge(nbuf, 1_c_int64_t, op /= K_SUM)))
      allocate(hc(merge(nbuf, 1_c_int64_t, op == K_MULADD)))
      call walk(ha, hb, hc)
    end if
    if (onheap) then
      s = pairwise(heapp, nb)
    else if (nb == 1) then
      s = stackp(1)
    else
      s = pairwise(stackp, nb)
    end if

  contains

    subroutine walk(ba, bb, bc)
      real(dp), intent(inout) :: ba(*), bb(*), bc(*)
      integer(c_int64_t) :: idx(nd), pa, pb, pc, j, m, pos, take
      integer :: k
      idx = 0
      pa = oa
      pb = ob
      pc = oc
      m = shape(1)
      pos = 0
      do
        j = 0
        do while (j < m)
          take = min(m - j, BLOCK - pos)
          call gather(ba, pos, a, pa + j*sa(1), sa(1), take)
          if (op /= K_SUM) call gather(bb, pos, b, pb + j*sb(1), sb(1), take)
          if (op == K_MULADD) call gather(bc, pos, c, pc + j*sc(1), sc(1), take)
          pos = pos + take
          j = j + take
          if (pos == BLOCK) then
            call store(blk(op, 1_c_int64_t, BLOCK, ba, bb, bc))
            pos = 0
          end if
        end do
        k = 2
        do while (k <= nd)
          idx(k) = idx(k) + 1
          pa = pa + sa(k)
          pb = pb + sb(k)
          pc = pc + sc(k)
          if (idx(k) < shape(k)) exit
          pa = pa - sa(k)*shape(k)
          pb = pb - sb(k)*shape(k)
          pc = pc - sc(k)*shape(k)
          idx(k) = 0
          k = k + 1
        end do
        if (k > nd) exit
      end do
      if (pos > 0) call store(blk(op, 1_c_int64_t, pos, ba, bb, bc))
    end subroutine walk

    subroutine walk_tiled(ba, bb, bc)
      real(dp), intent(inout) :: ba(*), bb(*), bc(*)
      real(dp), allocatable :: ta(:, :), tb(:, :), tc(:, :)
      integer(c_int64_t) :: idx(nd), pa, pb, pc, j, m, pos, take, j2, cnt, i, t
      integer :: k
      m = shape(1)
      allocate(ta(tt, m))
      allocate(tb(tt, merge(m, 1_c_int64_t, op /= K_SUM)))
      allocate(tc(tt, merge(m, 1_c_int64_t, op == K_MULADD)))
      idx = 0
      pa = oa
      pb = ob
      pc = oc
      pos = 0
      do
        do j2 = 0, shape(2) - 1, tt
          cnt = min(tt, shape(2) - j2)
          do i = 0, m - 1                      ! one memory row of the tile per logical element
            ta(1:cnt, i + 1) = a(pa + j2 + i*sa(1):pa + j2 + i*sa(1) + cnt - 1)
            if (op /= K_SUM) tb(1:cnt, i + 1) = b(pb + j2 + i*sb(1):pb + j2 + i*sb(1) + cnt - 1)
            if (op == K_MULADD) tc(1:cnt, i + 1) = c(pc + j2 + i*sc(1):pc + j2 + i*sc(1) + cnt - 1)
          end do
          do t = 1, cnt                        ! logical rows j2+1..j2+cnt, in order
            j = 0
            do while (j < m)
              take = min(m - j, BLOCK - pos)
              ba(pos + 1:pos + take) = ta(t, j + 1:j + take)
              if (op /= K_SUM) bb(pos + 1:pos + take) = tb(t, j + 1:j + take)
              if (op == K_MULADD) bc(pos + 1:pos + take) = tc(t, j + 1:j + take)
              pos = pos + take
              j = j + take
              if (pos == BLOCK) then
                call store(blk(op, 1_c_int64_t, BLOCK, ba, bb, bc))
                pos = 0
              end if
            end do
          end do
        end do
        k = 3
        do while (k <= nd)
          idx(k) = idx(k) + 1
          pa = pa + sa(k)
          pb = pb + sb(k)
          pc = pc + sc(k)
          if (idx(k) < shape(k)) exit
          pa = pa - sa(k)*shape(k)
          pb = pb - sb(k)*shape(k)
          pc = pc - sc(k)*shape(k)
          idx(k) = 0
          k = k + 1
        end do
        if (k > nd) exit
      end do
      if (pos > 0) call store(blk(op, 1_c_int64_t, pos, ba, bb, bc))
    end subroutine walk_tiled

    subroutine store(v)
      real(dp), intent(in) :: v
      nb = nb + 1
      if (onheap) then
        heapp(nb) = v
      else
        stackp(nb) = v
      end if
    end subroutine store

  end function seq_gather

  !> Strided-sequence reduction: picks the measured fastest of two bit-identical implementations
  !> (results/nd/RESULTS.md, round 2). Plain sums and large tileable sequences use the gather into
  !> vectorised block kernels; fused kernels on smaller or untileable sequences use the direct lane walk.
  function seq_reduce(op, nd, shape, a, oa, sa, b, ob, sb, c, oc, sc) result(s)
    integer, intent(in) :: op, nd
    integer(c_int64_t), intent(in) :: shape(nd), sa(nd), sb(nd), sc(nd), oa, ob, oc
    real(dp), intent(in) :: a(0:*), b(0:*), c(0:*)
    real(dp) :: s
    logical :: tiled
    tiled = .false.
    if (nd >= 2) tiled = product(shape) >= 1048576_c_int64_t .and. abs(sa(1)) > 8 .and. sa(2) == 1 .and. &
                         (op == K_SUM .or. sb(2) == 1) .and. (op /= K_MULADD .or. sc(2) == 1) .and. shape(2) > 1
    if (op /= K_SUM .and. .not. tiled) then
      s = seq_scalar(op, nd, shape, a, oa, sa, b, ob, sb, c, oc, sc)
    else
      s = seq_gather(op, nd, shape, a, oa, sa, b, ob, sb, c, oc, sc)
    end if
  end function seq_reduce

  !> Direct lane walk: each element goes straight into its lane accumulator. Bit-identical to reduce_blocks.
  function seq_scalar(op, nd, shape, a, oa, sa, b, ob, sb, c, oc, sc) result(s)
    integer, intent(in) :: op, nd
    integer(c_int64_t), intent(in) :: shape(nd), sa(nd), sb(nd), sc(nd), oa, ob, oc
    real(dp), intent(in) :: a(0:*), b(0:*), c(0:*)
    real(dp) :: s
    real(dp) :: acc(LANES), stackp(STACK_BLOCKS), t
    real(dp), allocatable :: heapp(:)
    integer(c_int64_t) :: n, nb, idx(nd), pa, pb, pc, j, m, pos
    integer :: lane, k
    logical :: onheap

    n = product(shape)
    if (n == 0) then
      s = 0.0_dp
      return
    end if
    onheap = (n + BLOCK - 1) / BLOCK > STACK_BLOCKS
    if (onheap) allocate(heapp((n + BLOCK - 1) / BLOCK))
    acc = 0.0_dp
    lane = 1
    pos = 0
    nb = 0
    idx = 0
    pa = oa
    pb = ob
    pc = oc
    m = shape(1)
    do
      select case (op)
      case (K_SUM)
        do j = 0, m - 1
          acc(lane) = acc(lane) + a(pa + j*sa(1))
          call advance()
        end do
      case (K_SQDIST)
        do j = 0, m - 1
          t = a(pa + j*sa(1)) - b(pb + j*sb(1))
          acc(lane) = acc(lane) + t*t
          call advance()
        end do
      case default   ! K_MULADD
        do j = 0, m - 1
          t = a(pa + j*sa(1))*b(pb + j*sb(1)) + c(pc + j*sc(1))
          acc(lane) = acc(lane) + t*t
          call advance()
        end do
      end select
      k = 2
      do while (k <= nd)
        idx(k) = idx(k) + 1
        pa = pa + sa(k)
        pb = pb + sb(k)
        pc = pc + sc(k)
        if (idx(k) < shape(k)) exit
        pa = pa - sa(k)*shape(k)
        pb = pb - sb(k)*shape(k)
        pc = pc - sc(k)*shape(k)
        idx(k) = 0
        k = k + 1
      end do
      if (k > nd) exit
    end do
    if (pos > 0) call flush()
    if (onheap) then
      s = pairwise(heapp, nb)
    else if (nb == 1) then
      s = stackp(1)
    else
      s = pairwise(stackp, nb)
    end if

  contains

    subroutine advance()
      lane = lane + 1
      if (lane > LANES) lane = 1
      pos = pos + 1
      if (pos == BLOCK) call flush()
    end subroutine advance

    subroutine flush()
      nb = nb + 1
      if (onheap) then
        heapp(nb) = lane_tree(acc)
      else
        stackp(nb) = lane_tree(acc)
      end if
      acc = 0.0_dp
      lane = 1
      pos = 0
    end subroutine flush

  end function seq_scalar

  !> buf(pos+1 : pos+take) = x(start), x(start+stride), ... (stride may be negative or zero).
  subroutine gather(buf, pos, x, start, stride, take)
    real(dp), intent(inout) :: buf(*)
    real(dp), intent(in) :: x(0:*)
    integer(c_int64_t), intent(in) :: pos, start, stride, take
    integer(c_int64_t) :: t
    if (stride == 1) then
      buf(pos + 1:pos + take) = x(start:start + take - 1)
    else if (stride == 0) then
      buf(pos + 1:pos + take) = x(start)
    else
      do t = 0, take - 1
        buf(pos + 1 + t) = x(start + t*stride)
      end do
    end if
  end subroutine gather

  !> Column sums: out(po + j*so) = blocked sum over i = 0..K-1 of x(px + j + i*sr), j = 0..w-1 (w <= WIDE).
  !> Lane accumulators are kept per column, so each column's order equals reduce_blocks on that column.
  subroutine sum_cols(K, sr, w, x, px, out, po, so)
    integer(c_int64_t), intent(in) :: K, sr, w, px, po, so
    real(dp), intent(in) :: x(0:*)
    real(dp), intent(inout) :: out(0:*)
    real(dp), allocatable :: part(:, :), acc(:, :)
    integer(c_int64_t) :: nb, kb, j

    allocate(acc(w, LANES))
    nb = (K + BLOCK - 1) / BLOCK
    if (nb == 1) then
      call cols_block(0_c_int64_t, K, sr, w, x, px, out, po, so, acc)
      return
    end if
    allocate(part(nb, w))
    do kb = 1, nb
      ! block kb's per-column results go to part(kb, :) (stride nb)
      call cols_block((kb - 1)*BLOCK, min(K, kb*BLOCK), sr, w, x, px, part(kb, 1), 0_c_int64_t, nb, acc)   ! element: sequence association with the rest of part
    end do
    do j = 1, w
      out(po + (j - 1)*so) = pairwise(part(:, j), nb)
    end do
  end subroutine sum_cols

  !> One block of rows [i0, i1) for w columns: res(ro + j*rs) = lane_tree of the block's lane sums of column j.
  subroutine cols_block(i0, i1, sr, w, x, px, res, ro, rs, acc)
    integer(c_int64_t), intent(in) :: i0, i1, sr, w, px, ro, rs
    real(dp), intent(in) :: x(0:*)
    real(dp), intent(inout) :: res(0:*)
    real(dp), intent(out) :: acc(w, LANES)
    real(dp) :: lv(LANES)
    integer(c_int64_t) :: i, base, j
    integer :: lane
    acc(1:w, :) = 0.0_dp
    lane = 1
    do i = i0, i1 - 1
      base = px + i*sr
      acc(1:w, lane) = acc(1:w, lane) + x(base:base + w - 1)
      lane = lane + 1
      if (lane > LANES) lane = 1
    end do
    do j = 1, w
      lv = acc(j, :)
      res(ro + (j - 1)*rs) = lane_tree(lv)
    end do
  end subroutine cols_block

  ! ----------------------------------------------------------------------------------------------
  ! C ABI
  ! ----------------------------------------------------------------------------------------------

  !> Blocked reduction of f over the C-order sequence of n-d strided operands of identical shape.
  !> desc(nd, 4): extents, then element strides of a, b, c. op: K_SQDIST, K_MULADD or K_SUM (b, c unused).
  function numj_reduce_seq(op, nd, desc, a, oa, b, ob, c, oc) bind(C, name='numj_reduce_seq') result(s)
    integer(c_int), value :: op, nd
    integer(c_int64_t), intent(in) :: desc(nd, 4)
    integer(c_int64_t), value :: oa, ob, oc
    real(c_double), intent(in) :: a(0:*), b(0:*), c(0:*)
    real(c_double) :: s
    s = seq_reduce(int(op), int(nd), desc(:, 1), a, oa, desc(:, 2), b, ob, desc(:, 3), c, oc, desc(:, 4))
  end function numj_reduce_seq

  !> out[kept] = blocked sum over the reduced axes of x.
  !> desc = [shapek(nk), sxk(nk), sok(nk), shaper(nr), sxr(nr)]: kept extents with x and out strides, then
  !> reduced extents (C order of the reduced axes, dim 1 innermost) with x strides. Requires nr >= 1, every
  !> reduced extent >= 1 and every kept extent >= 1 (the Java layer handles the empty cases).
  !> Paths: (1) one reduced axis with unit stride: contiguous sequences via reduce_blocks;
  !>        (2) one reduced axis and a unit-stride kept axis: column tiles (sum_cols);
  !>        (3) anything else: general strided walk (seq_reduce). All three give identical bits.
  subroutine numj_sum_nd(nk, nr, desc, x, ox, out, oo, nthreads) bind(C, name='numj_sum_nd')
    integer(c_int), value :: nk, nr, nthreads
    integer(c_int64_t), intent(in) :: desc(*)
    integer(c_int64_t), value :: ox, oo
    real(c_double), intent(in) :: x(0:*)
    real(c_double), intent(inout) :: out(0:*)
    integer(c_int64_t) :: shapek(nk), sxk(nk), sok(nk), shaper(nr), sxr(nr)
    integer(c_int64_t) :: nout, K, nwork, chunk, t, r0, r1, ntile, tw
    integer :: path

    ntile = 1
    tw = TILE

    shapek = desc(1:nk)
    sxk = desc(nk + 1:2*nk)
    sok = desc(2*nk + 1:3*nk)
    shaper = desc(3*nk + 1:3*nk + nr)
    sxr = desc(3*nk + nr + 1:3*nk + 2*nr)
    nout = product(shapek)
    K = product(shaper)

    if (nr == 1 .and. sxr(1) == 1) then
      path = 1
      if (nout == 1) then          ! one long sequence: parallelism inside reduce_blocks
        out(oo) = reduce_blocks(K_SUM, K, x(ox), x(ox), x(ox), nthreads)
        return
      end if
      nwork = nout
    else if (nr == 1 .and. nk >= 1 .and. sxk(1) == 1) then
      path = 2
      tw = min(WIDE, shapek(1), max(TILE, (shapek(1) + nthreads - 1) / max(nthreads, 1)))
      ntile = (shapek(1) + tw - 1) / tw
      nwork = (nout / shapek(1)) * ntile
    else
      path = 3
      nwork = nout
    end if

    if (nthreads > 1 .and. nwork > 1) then
      chunk = (nwork + nthreads - 1) / nthreads
      !$omp parallel do num_threads(nthreads) schedule(static, 1) private(r0, r1)
      do t = 0, nthreads - 1
        r0 = t*chunk
        r1 = min(nwork, r0 + chunk) - 1
        if (r0 <= r1) call work(r0, r1)
      end do
      !$omp end parallel do
    else
      call work(0_c_int64_t, nwork - 1)
    end if

  contains

    subroutine work(w0, w1)
      integer(c_int64_t), intent(in) :: w0, w1
      integer(c_int64_t) :: w, px, po, ro, j0
      real(dp) :: acc(LANES)
      if (path == 1 .and. nk == 1) then
        ! Rows of one kept axis: offsets by multiplication (no div/mod unravel per output). Rows of at most
        ! LANES elements fill lanes 1..K once, exactly as blk_sum's tail does (acc = 0 + x), so the bits match.
        if (K <= 8) then
          call short_rows(w0, w1)
        else if (K <= LANES) then
          do w = w0, w1
            px = ox + w*sxk(1)
            acc = 0.0_dp
            acc(1:K) = acc(1:K) + x(px:px + K - 1)
            out(oo + w*sok(1)) = lane_tree(acc)
          end do
        else
          do w = w0, w1
            px = ox + w*sxk(1)
            out(oo + w*sok(1)) = reduce_blocks(K_SUM, K, x(px), x(px), x(px), 1_c_int)
          end do
        end if
        return
      end if
      do w = w0, w1
        select case (path)
        case (1)
          px = ox + unravel_off(w, int(nk), shapek, sxk)
          po = oo + unravel_off(w, int(nk), shapek, sok)
          out(po) = reduce_blocks(K_SUM, K, x(px), x(px), x(px), 1_c_int)
        case (2)
          ro = w / ntile
          j0 = mod(w, ntile) * tw
          px = ox + unravel_off(ro, int(nk) - 1, shapek(2:), sxk(2:)) + j0
          po = oo + unravel_off(ro, int(nk) - 1, shapek(2:), sok(2:)) + j0*sok(1)
          call sum_cols(K, sxr(1), min(tw, shapek(1) - j0), x, px, out, po, sok(1))
        case default
          px = ox + unravel_off(w, int(nk), shapek, sxk)
          po = oo + unravel_off(w, int(nk), shapek, sok)
          out(po) = seq_reduce(K_SUM, int(nr), shaper, x, px, sxr, x, px, sxr, x, px, sxr)
        end select
      end do
    end subroutine work

    !> Rows of K <= 8 contiguous elements. With lanes a(l) = 0 + x(l) (l <= K) and +0.0 elsewhere, lane_tree
    !> reduces exactly to ((a1+a5) + (a3+a7)) + ((a2+a6) + (a4+a8)): adding +0.0 to a lane value is exact
    !> because a lane that starts at +0.0 can never hold -0.0. One loop per K, no lane array, no calls.
    subroutine short_rows(w0, w1)
      integer(c_int64_t), intent(in) :: w0, w1
      integer(c_int64_t) :: w, p, rs, os
      real(dp), parameter :: z = 0.0_dp
      rs = sxk(1)
      os = sok(1)
      select case (int(K))
      case (1)
        do w = w0, w1
          p = ox + w*rs
          out(oo + w*os) = z + x(p)
        end do
      case (2)
        do w = w0, w1
          p = ox + w*rs
          out(oo + w*os) = (z + x(p)) + (z + x(p + 1))
        end do
      case (3)
        do w = w0, w1
          p = ox + w*rs
          out(oo + w*os) = ((z + x(p)) + (z + x(p + 2))) + (z + x(p + 1))
        end do
      case (4)
        do w = w0, w1
          p = ox + w*rs
          out(oo + w*os) = ((z + x(p)) + (z + x(p + 2))) + ((z + x(p + 1)) + (z + x(p + 3)))
        end do
      case (5)
        do w = w0, w1
          p = ox + w*rs
          out(oo + w*os) = (((z + x(p)) + (z + x(p + 4))) + (z + x(p + 2))) + ((z + x(p + 1)) + (z + x(p + 3)))
        end do
      case (6)
        do w = w0, w1
          p = ox + w*rs
          out(oo + w*os) = (((z + x(p)) + (z + x(p + 4))) + (z + x(p + 2))) &
                         + (((z + x(p + 1)) + (z + x(p + 5))) + (z + x(p + 3)))
        end do
      case (7)
        do w = w0, w1
          p = ox + w*rs
          out(oo + w*os) = (((z + x(p)) + (z + x(p + 4))) + ((z + x(p + 2)) + (z + x(p + 6)))) &
                         + (((z + x(p + 1)) + (z + x(p + 5))) + (z + x(p + 3)))
        end do
      case default
        do w = w0, w1
          p = ox + w*rs
          out(oo + w*os) = (((z + x(p)) + (z + x(p + 4))) + ((z + x(p + 2)) + (z + x(p + 6)))) &
                         + (((z + x(p + 1)) + (z + x(p + 5))) + ((z + x(p + 3)) + (z + x(p + 7))))
        end do
      end select
    end subroutine short_rows

  end subroutine numj_sum_nd

end module numj_reduce

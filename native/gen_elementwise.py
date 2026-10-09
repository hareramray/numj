"""Generates native/numj_elementwise.f90 (float64 elementwise kernels).

The generated file is committed; the build does not need Python. Re-run after editing this script:
    python -I native/gen_elementwise.py

Why a generator: every (operation x operand-layout) combination needs its own loop so that the operation is a
compile-time constant inside the loop (vectorisable, no per-element dispatch). Writing ~80 near-identical loops by
hand is error-prone; adding float32 later means emitting the same templates for another kind.

Aliasing contract (Fortran forbids modifying a dummy argument through another dummy):
  * out-of-place routines receive outputs that are disjoint from every input;
  * in-place routines receive the modified array exactly once ("x"); the other operand is disjoint from it;
  * x = x op x uses the dedicated *_self routines.
The Java layer guarantees this (identical-layout operands become in-place calls, any other overlap is resolved by
copying the input first).
"""
import pathlib

OPS = [  # code, name, expression in terms of {x} (left / modified operand) and {y}
    (1, "ADD", "{x} + {y}"),
    (2, "SUB", "{x} - {y}"),
    (3, "MUL", "{x} * {y}"),
    (4, "DIV", "{x} / {y}"),
    (5, "RSUB", "{y} - {x}"),
    (6, "RDIV", "{y} / {x}"),
]
SELF_OPS = OPS[:4]


def cases(ops, body, indent):
    pad = " " * indent
    out = []
    for code, name, expr in ops:
        out.append(f"{pad}case (OP_{name})")
        for line in body(expr):
            out.append(f"{pad}  {line}")
    out.append(f"{pad}case default")
    out.append(f"{pad}  error stop 'numj: bad elementwise op'")
    return "\n".join(out)


def contiguous_kernels():
    """Explicit-shape dummies: contiguous, and (by the contract above) non-aliasing."""
    vv = cases(OPS, lambda e: [f"out = {e.format(x='a', y='b')}"], 4)
    vs = cases(OPS, lambda e: [f"out = {e.format(x='a', y='s')}"], 4)
    ipv = cases(OPS, lambda e: [f"x = {e.format(x='x', y='y')}"], 4)
    ips = cases(OPS, lambda e: [f"x = {e.format(x='x', y='s')}"], 4)
    slf = cases(SELF_OPS, lambda e: [f"x = {e.format(x='x', y='x')}"], 4)
    return f"""
  !> out = a op b
  subroutine ew_vv(op, m, a, b, out)
    integer(c_int), intent(in) :: op
    integer(c_int64_t), intent(in) :: m
    real(dp), intent(in) :: a(m), b(m)
    real(dp), intent(out) :: out(m)
    select case (op)
{vv}
    end select
  end subroutine ew_vv

  !> out = a op s
  subroutine ew_vs(op, m, a, s, out)
    integer(c_int), intent(in) :: op
    integer(c_int64_t), intent(in) :: m
    real(dp), intent(in) :: a(m), s
    real(dp), intent(out) :: out(m)
    select case (op)
{vs}
    end select
  end subroutine ew_vs

  !> x = x op y
  subroutine ew_ip_vv(op, m, x, y)
    integer(c_int), intent(in) :: op
    integer(c_int64_t), intent(in) :: m
    real(dp), intent(inout) :: x(m)
    real(dp), intent(in) :: y(m)
    select case (op)
{ipv}
    end select
  end subroutine ew_ip_vv

  !> x = x op s
  subroutine ew_ip_vs(op, m, x, s)
    integer(c_int), intent(in) :: op
    integer(c_int64_t), intent(in) :: m
    real(dp), intent(inout) :: x(m)
    real(dp), intent(in) :: s
    select case (op)
{ips}
    end select
  end subroutine ew_ip_vs

  !> x = x op x
  subroutine ew_self(op, m, x)
    integer(c_int), intent(in) :: op
    integer(c_int64_t), intent(in) :: m
    real(dp), intent(inout) :: x(m)
    select case (op)
{slf}
    end select
  end subroutine ew_self
"""


def strided_kernels():
    """General strided rows: element j of a row is at base + j*stride (strides may be negative or zero)."""
    def loop(stmt):
        return ["do j = 0, m - 1", f"  {stmt}", "end do"]
    oop = cases(OPS, lambda e: loop(f"out(io + j*ko) = {e.format(x='a(ia + j*ka)', y='b(ib + j*kb)')}"), 4)
    ip = cases(OPS, lambda e: loop(f"x(ix + j*kx) = {e.format(x='x(ix + j*kx)', y='y(iy + j*ky)')}"), 4)
    slf = cases(SELF_OPS, lambda e: loop(f"x(ix + j*kx) = {e.format(x='x(ix + j*kx)', y='x(ix + j*kx)')}"), 4)
    return f"""
  subroutine ew_strided(op, m, out, io, ko, a, ia, ka, b, ib, kb)
    integer(c_int), intent(in) :: op
    integer(c_int64_t), intent(in) :: m, io, ko, ia, ka, ib, kb
    real(dp), intent(inout) :: out(0:*)
    real(dp), intent(in) :: a(0:*), b(0:*)
    integer(c_int64_t) :: j
    select case (op)
{oop}
    end select
  end subroutine ew_strided

  subroutine ew_ip_strided(op, m, x, ix, kx, y, iy, ky)
    integer(c_int), intent(in) :: op
    integer(c_int64_t), intent(in) :: m, ix, kx, iy, ky
    real(dp), intent(inout) :: x(0:*)
    real(dp), intent(in) :: y(0:*)
    integer(c_int64_t) :: j
    select case (op)
{ip}
    end select
  end subroutine ew_ip_strided

  subroutine ew_self_strided(op, m, x, ix, kx)
    integer(c_int), intent(in) :: op
    integer(c_int64_t), intent(in) :: m, ix, kx
    real(dp), intent(inout) :: x(0:*)
    integer(c_int64_t) :: j
    select case (op)
{slf}
    end select
  end subroutine ew_self_strided
"""


def nd_driver(name, cname, doc, operands, extra_args, extra_decls, row_call):
    """n-d driver: rows along dim 1 (innermost), outer dims walked by an odometer, optional OpenMP over rows.

    operands: list of (array, desc column, offset arg) in descriptor order after the shape column.
    """
    k = len(operands)
    arrays = ", ".join(f"{a}, o{a}" for a, _, _ in operands)
    offs = ", ".join(f"p{a}" for a, _, _ in operands)
    init = "\n".join(f"      p{a} = o{a} + offset_of(r0, nd, desc(:, 1), desc(:, {col}))" for a, col, _ in operands)
    step = "\n".join(f"          p{a} = p{a} + desc(kk, {col})" for a, col, _ in operands)
    back = "\n".join(f"          p{a} = p{a} - desc(kk, {col}) * desc(kk, 1)" for a, col, _ in operands)
    decls = "\n".join(extra_decls)
    return f"""
  !> {doc}
  !> desc(nd, {k + 1}): column 1 = extents, columns 2.. = element strides of {", ".join(a for a, _, _ in operands)};
  !> dim 1 is the innermost (fastest) axis. o* = element offset of the first logical element from the pointer.
  subroutine {name}({extra_args}nd, desc, {arrays}, nthreads) bind(C, name='{cname}')
    integer(c_int), value :: nd, nthreads
    integer(c_int64_t), intent(in) :: desc(nd, {k + 1})
    integer(c_int64_t), value :: {", ".join(f"o{a}" for a, _, _ in operands)}
{decls}
    integer(c_int64_t) :: nrows, r0, r1, chunk, t
    nrows = 1
    if (nd > 1) nrows = product(desc(2:nd, 1))
    if (nthreads > 1 .and. nrows > 1) then
      chunk = (nrows + nthreads - 1) / nthreads
      !$omp parallel do num_threads(nthreads) schedule(static, 1) private(r0, r1)
      do t = 0, nthreads - 1
        r0 = t * chunk
        r1 = min(nrows, r0 + chunk) - 1
        if (r0 <= r1) call rows(r0, r1)
      end do
      !$omp end parallel do
    else
      call rows(0_c_int64_t, nrows - 1)
    end if

  contains

    subroutine rows(r0, r1)
      integer(c_int64_t), intent(in) :: r0, r1
      integer(c_int64_t) :: r, {offs}, idx(nd)
      integer :: kk
{init}
      call unravel_index(r0, nd, desc(:, 1), idx)
      do r = r0, r1
        {row_call}
        kk = 2
        do while (kk <= nd)
          idx(kk) = idx(kk) + 1
{step}
          if (idx(kk) < desc(kk, 1)) exit
{back}
          idx(kk) = 0
          kk = kk + 1
        end do
      end do
    end subroutine rows

  end subroutine {name}
"""


def main():
    oop = nd_driver(
        "numj_ew_nd", "numj_ew_nd", "out = a op b over n-d strided operands (out disjoint from a and b).",
        [("out", 2, None), ("a", 3, None), ("b", 4, None)], "op, ", [
            "    integer(c_int), value :: op",
            "    real(c_double), intent(inout) :: out(0:*)",
            "    real(c_double), intent(in) :: a(0:*), b(0:*)"],
        "call row_oop(op, desc(1, 1), out, pout, desc(1, 2), a, pa, desc(1, 3), b, pb, desc(1, 4))")
    ip = nd_driver(
        "numj_ew_nd_ip", "numj_ew_nd_ip", "x = x op y over n-d strided operands (y disjoint from x).",
        [("x", 2, None), ("y", 3, None)], "op, ", [
            "    integer(c_int), value :: op",
            "    real(c_double), intent(inout) :: x(0:*)",
            "    real(c_double), intent(in) :: y(0:*)"],
        "call row_ip(op, desc(1, 1), x, px, desc(1, 2), y, py, desc(1, 3))")
    slf = nd_driver(
        "numj_ew_nd_self", "numj_ew_nd_self", "x = x op x over an n-d strided operand.",
        [("x", 2, None)], "op, ", [
            "    integer(c_int), value :: op",
            "    real(c_double), intent(inout) :: x(0:*)"],
        "call row_self(op, desc(1, 1), x, px, desc(1, 2))")
    cpy = nd_driver(
        "numj_copy_nd", "numj_copy_nd", "out = a (bitwise copy) over n-d strided operands (out disjoint from a).",
        [("out", 2, None), ("a", 3, None)], "", [
            "    real(c_double), intent(inout) :: out(0:*)",
            "    real(c_double), intent(in) :: a(0:*)"],
        "call row_copy(desc(1, 1), out, pout, desc(1, 2), a, pa, desc(1, 3))")

    src = f"""!> GENERATED by native/gen_elementwise.py -- do not edit by hand.
!>
!> numj elementwise kernels (float64): out = a op b, scalar operands, in-place updates, bitwise copies.
!> Contiguous entry points take plain pointers; n-d entry points take a descriptor of extents and element
!> strides (any sign, zero for broadcast operands) and walk rows along the innermost axis. Each row is
!> dispatched to a unit-stride kernel when its strides allow (the common case after the Java layer coalesces
!> axes) and to a general strided loop otherwise.
!>
!> Every result element is a single correctly rounded IEEE-754 operation (built with -ffp-contract=off), so
!> results are bitwise identical to Java's and NumPy's for any layout, path or thread count.
!> Aliasing contract: see gen_elementwise.py. Bounds and lifetimes are validated by the Java layer.
module numj_elementwise
  use, intrinsic :: iso_c_binding, only: c_double, c_int, c_int64_t
  implicit none
  private

  integer, parameter :: dp = c_double
{chr(10).join(f"  integer(c_int), parameter :: OP_{n} = {c}" for c, n, _ in OPS)}
  !> Elements per OpenMP work item in the contiguous kernels.
  integer(c_int64_t), parameter :: CHUNK = 32768_c_int64_t

  public :: numj_ew_contig, numj_ew_contig_s, numj_ew_contig_ip, numj_ew_contig_ip_s, numj_ew_contig_self, &
            numj_ew_nd, numj_ew_nd_ip, numj_ew_nd_self, numj_copy_nd

contains

  ! ------------------------------------------------------------------------------------------------
  ! Unit-stride kernels
  ! ------------------------------------------------------------------------------------------------
{contiguous_kernels()}
  ! ------------------------------------------------------------------------------------------------
  ! General strided rows
  ! ------------------------------------------------------------------------------------------------
{strided_kernels()}
  ! ------------------------------------------------------------------------------------------------
  ! Row dispatch: pick the unit-stride kernel when the row allows it (sequence association passes the
  ! row's first element; no copies). Swapping operands uses the reversed op (x - y == -(y - x) is NOT used;
  ! RSUB/RDIV evaluate y - x and y / x directly, so every result is the same single rounding).
  ! ------------------------------------------------------------------------------------------------

  pure function reversed(op) result(r)
    integer(c_int), intent(in) :: op
    integer(c_int) :: r
    select case (op)
    case (OP_SUB); r = OP_RSUB
    case (OP_DIV); r = OP_RDIV
    case (OP_RSUB); r = OP_SUB
    case (OP_RDIV); r = OP_DIV
    case default; r = op
    end select
  end function reversed

  subroutine row_oop(op, m, out, io, ko, a, ia, ka, b, ib, kb)
    integer(c_int), intent(in) :: op
    integer(c_int64_t), intent(in) :: m, io, ko, ia, ka, ib, kb
    real(dp), intent(inout) :: out(0:*)
    real(dp), intent(in) :: a(0:*), b(0:*)
    if (ko == 1 .and. ka == 1 .and. kb == 1) then
      call ew_vv(op, m, a(ia), b(ib), out(io))
    else if (ko == 1 .and. ka == 1 .and. kb == 0) then
      call ew_vs(op, m, a(ia), b(ib), out(io))
    else if (ko == 1 .and. ka == 0 .and. kb == 1) then
      call ew_vs(reversed(op), m, b(ib), a(ia), out(io))
    else
      call ew_strided(op, m, out, io, ko, a, ia, ka, b, ib, kb)
    end if
  end subroutine row_oop

  subroutine row_ip(op, m, x, ix, kx, y, iy, ky)
    integer(c_int), intent(in) :: op
    integer(c_int64_t), intent(in) :: m, ix, kx, iy, ky
    real(dp), intent(inout) :: x(0:*)
    real(dp), intent(in) :: y(0:*)
    if (kx == 1 .and. ky == 1) then
      call ew_ip_vv(op, m, x(ix), y(iy))
    else if (kx == 1 .and. ky == 0) then
      call ew_ip_vs(op, m, x(ix), y(iy))
    else
      call ew_ip_strided(op, m, x, ix, kx, y, iy, ky)
    end if
  end subroutine row_ip

  subroutine row_self(op, m, x, ix, kx)
    integer(c_int), intent(in) :: op
    integer(c_int64_t), intent(in) :: m, ix, kx
    real(dp), intent(inout) :: x(0:*)
    if (kx == 1) then
      call ew_self(op, m, x(ix))
    else
      call ew_self_strided(op, m, x, ix, kx)
    end if
  end subroutine row_self

  subroutine row_copy(m, out, io, ko, a, ia, ka)
    integer(c_int64_t), intent(in) :: m, io, ko, ia, ka
    real(dp), intent(inout) :: out(0:*)
    real(dp), intent(in) :: a(0:*)
    integer(c_int64_t) :: j
    if (ko == 1 .and. ka == 1) then
      out(io:io + m - 1) = a(ia:ia + m - 1)
    else if (ko == 1 .and. ka == 0) then
      out(io:io + m - 1) = a(ia)
    else
      do j = 0, m - 1
        out(io + j*ko) = a(ia + j*ka)
      end do
    end if
  end subroutine row_copy

  ! ------------------------------------------------------------------------------------------------
  ! Index helpers (dim 1 innermost; "rows" enumerate dims 2..nd in C order)
  ! ------------------------------------------------------------------------------------------------

  !> Odometer state (idx(2:nd)) of row r.
  pure subroutine unravel_index(r, nd, shape, idx)
    integer(c_int64_t), intent(in) :: r
    integer(c_int), intent(in) :: nd
    integer(c_int64_t), intent(in) :: shape(nd)
    integer(c_int64_t), intent(out) :: idx(nd)
    integer(c_int64_t) :: q
    integer :: k
    idx = 0
    q = r
    do k = 2, nd
      idx(k) = mod(q, shape(k))
      q = q / shape(k)
    end do
  end subroutine unravel_index

  !> Element offset of the first element of row r for one operand.
  pure function offset_of(r, nd, shape, stride) result(off)
    integer(c_int64_t), intent(in) :: r
    integer(c_int), intent(in) :: nd
    integer(c_int64_t), intent(in) :: shape(nd), stride(nd)
    integer(c_int64_t) :: off, q
    integer :: k
    off = 0
    q = r
    do k = 2, nd
      off = off + mod(q, shape(k)) * stride(k)
      q = q / shape(k)
    end do
  end function offset_of

  ! ------------------------------------------------------------------------------------------------
  ! C ABI: contiguous
  ! ------------------------------------------------------------------------------------------------

  !> out(1:n) = a op b. out is disjoint from a and b.
  subroutine numj_ew_contig(op, n, a, b, out, nthreads) bind(C, name='numj_ew_contig')
    integer(c_int), value :: op, nthreads
    integer(c_int64_t), value :: n
    real(c_double), intent(in) :: a(n), b(n)
    real(c_double), intent(out) :: out(n)
    integer(c_int64_t) :: k, lo, hi
    if (nthreads > 1 .and. n > CHUNK) then
      !$omp parallel do num_threads(nthreads) schedule(static) private(lo, hi)
      do k = 0, (n - 1) / CHUNK
        lo = k*CHUNK + 1
        hi = min(n, lo + CHUNK - 1)
        call ew_vv(op, hi - lo + 1, a(lo), b(lo), out(lo))
      end do
      !$omp end parallel do
    else
      call ew_vv(op, n, a, b, out)
    end if
  end subroutine numj_ew_contig

  !> out(1:n) = a op s. out is disjoint from a.
  subroutine numj_ew_contig_s(op, n, a, s, out, nthreads) bind(C, name='numj_ew_contig_s')
    integer(c_int), value :: op, nthreads
    integer(c_int64_t), value :: n
    real(c_double), value :: s
    real(c_double), intent(in) :: a(n)
    real(c_double), intent(out) :: out(n)
    integer(c_int64_t) :: k, lo, hi
    if (nthreads > 1 .and. n > CHUNK) then
      !$omp parallel do num_threads(nthreads) schedule(static) private(lo, hi)
      do k = 0, (n - 1) / CHUNK
        lo = k*CHUNK + 1
        hi = min(n, lo + CHUNK - 1)
        call ew_vs(op, hi - lo + 1, a(lo), s, out(lo))
      end do
      !$omp end parallel do
    else
      call ew_vs(op, n, a, s, out)
    end if
  end subroutine numj_ew_contig_s

  !> x(1:n) = x op y. y is disjoint from x.
  subroutine numj_ew_contig_ip(op, n, x, y, nthreads) bind(C, name='numj_ew_contig_ip')
    integer(c_int), value :: op, nthreads
    integer(c_int64_t), value :: n
    real(c_double), intent(inout) :: x(n)
    real(c_double), intent(in) :: y(n)
    integer(c_int64_t) :: k, lo, hi
    if (nthreads > 1 .and. n > CHUNK) then
      !$omp parallel do num_threads(nthreads) schedule(static) private(lo, hi)
      do k = 0, (n - 1) / CHUNK
        lo = k*CHUNK + 1
        hi = min(n, lo + CHUNK - 1)
        call ew_ip_vv(op, hi - lo + 1, x(lo), y(lo))
      end do
      !$omp end parallel do
    else
      call ew_ip_vv(op, n, x, y)
    end if
  end subroutine numj_ew_contig_ip

  !> x(1:n) = x op s.
  subroutine numj_ew_contig_ip_s(op, n, x, s, nthreads) bind(C, name='numj_ew_contig_ip_s')
    integer(c_int), value :: op, nthreads
    integer(c_int64_t), value :: n
    real(c_double), value :: s
    real(c_double), intent(inout) :: x(n)
    integer(c_int64_t) :: k, lo, hi
    if (nthreads > 1 .and. n > CHUNK) then
      !$omp parallel do num_threads(nthreads) schedule(static) private(lo, hi)
      do k = 0, (n - 1) / CHUNK
        lo = k*CHUNK + 1
        hi = min(n, lo + CHUNK - 1)
        call ew_ip_vs(op, hi - lo + 1, x(lo), s)
      end do
      !$omp end parallel do
    else
      call ew_ip_vs(op, n, x, s)
    end if
  end subroutine numj_ew_contig_ip_s

  !> x(1:n) = x op x.
  subroutine numj_ew_contig_self(op, n, x, nthreads) bind(C, name='numj_ew_contig_self')
    integer(c_int), value :: op, nthreads
    integer(c_int64_t), value :: n
    real(c_double), intent(inout) :: x(n)
    integer(c_int64_t) :: k, lo, hi
    if (nthreads > 1 .and. n > CHUNK) then
      !$omp parallel do num_threads(nthreads) schedule(static) private(lo, hi)
      do k = 0, (n - 1) / CHUNK
        lo = k*CHUNK + 1
        hi = min(n, lo + CHUNK - 1)
        call ew_self(op, hi - lo + 1, x(lo))
      end do
      !$omp end parallel do
    else
      call ew_self(op, n, x)
    end if
  end subroutine numj_ew_contig_self

  ! ------------------------------------------------------------------------------------------------
  ! C ABI: n-d strided
  ! ------------------------------------------------------------------------------------------------
{oop}{ip}{slf}{cpy}
end module numj_elementwise
"""
    out = pathlib.Path(__file__).with_name("numj_elementwise.f90")
    out.write_text(src, encoding="utf-8", newline="\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

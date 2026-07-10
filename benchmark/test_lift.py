import pytest
import torch

from . import base, consts


@pytest.mark.lift
def test_lift():
    bench = base.UnaryPointwiseBenchmark(
        op_name="lift",
        torch_op=torch.ops.aten.lift,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.lift_out
def test_lift_out():
    # The `out` argument is created and injected by UnaryPointwiseOutBenchmark.
    # Its get_input_iter() yields (inp, {"out": torch.empty_like(inp)}) for each
    # shape, and Benchmark.run() forwards it as kwargs, so the actual call is
    # torch.ops.aten.lift.out(inp, out=out).
    bench = base.UnaryPointwiseOutBenchmark(
        op_name="lift_out",
        torch_op=torch.ops.aten.lift.out,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()

"""DEVSIM loads the shim as its only math library and its solves call through it."""

import devsim as ds
import numpy as np
import pytest

import devsim_openblas


def _laplace(device, region):
    ds.edge_from_node_model(device=device, region=region, node_model="V")
    ds.edge_model(device=device, region=region, name="E",
                  equation="(V@n0 - V@n1) * EdgeInverseLength")
    ds.edge_model(device=device, region=region, name="E:V@n0", equation="EdgeInverseLength")
    ds.edge_model(device=device, region=region, name="E:V@n1", equation="-EdgeInverseLength")


def test_shim_is_the_math_library():
    info = ds.get_parameter(name="info")
    print("DEVSIM info:", info)
    assert list(info["math_libraries"]) == [str(devsim_openblas.shim_path())]


@pytest.fixture(scope="module")
def resistor():
    """1D Laplace problem, V = 1 at x = 0 and V = 0 at x = 1."""
    ds.create_1d_mesh(mesh="m1")
    ds.add_1d_mesh_line(mesh="m1", pos=0.0, ps=0.05, tag="left")
    ds.add_1d_mesh_line(mesh="m1", pos=1.0, ps=0.05, tag="right")
    ds.add_1d_contact(mesh="m1", name="left", tag="left", material="metal")
    ds.add_1d_contact(mesh="m1", name="right", tag="right", material="metal")
    ds.add_1d_region(mesh="m1", material="Si", region="r", tag1="left", tag2="right")
    ds.finalize_mesh(mesh="m1")
    ds.create_device(mesh="m1", device="d1")
    ds.node_solution(device="d1", region="r", name="V")
    _laplace("d1", "r")
    ds.equation(device="d1", region="r", name="VEq", variable_name="V",
                edge_model="E", variable_update="default")
    for contact, bias in (("left", 1.0), ("right", 0.0)):
        ds.set_parameter(device="d1", name=f"{contact}_bias", value=bias)
        ds.contact_node_model(device="d1", contact=contact, name=f"{contact}_bc",
                              equation=f"V - {contact}_bias")
        ds.contact_node_model(device="d1", contact=contact, name=f"{contact}_bc:V",
                              equation="1")
        ds.contact_equation(device="d1", contact=contact, name="VEq",
                            node_model=f"{contact}_bc")
    return "d1"


def _solve_and_check(device, **solver):
    ds.set_node_values(device=device, region="r", name="V", init_from="x")
    before = devsim_openblas.calls()
    ds.solve(type="dc", absolute_error=1e-12, relative_error=1e-12,
             maximum_iterations=20, **solver)
    after = devsim_openblas.calls()
    x = np.array(ds.get_node_model_values(device=device, region="r", name="x"))
    v = np.array(ds.get_node_model_values(device=device, region="r", name="V"))
    np.testing.assert_allclose(v, 1.0 - x, atol=1e-9)
    called = {name: after[name] - before[name] for name in after if after[name] > before[name]}
    print("shim calls:", called)
    return called


def test_direct_dc_solve(resistor):
    print("direct solver:", ds.get_parameter(name="direct_solver"))
    assert _solve_and_check(resistor), "direct solve called no BLAS routine"


def test_iterative_dc_solve_uses_drotg(resistor):
    assert "drotg" in _solve_and_check(resistor, solver_type="iterative")


def test_element_field_uses_dgetrf_dgetrs():
    ds.create_2d_mesh(mesh="m2")
    for direction in ("x", "y"):
        ds.add_2d_mesh_line(mesh="m2", dir=direction, pos=0.0, ps=0.25)
        ds.add_2d_mesh_line(mesh="m2", dir=direction, pos=1.0, ps=0.25)
    ds.add_2d_region(mesh="m2", material="Si", region="r", xl=0.0, xh=1.0, yl=0.0, yh=1.0)
    ds.finalize_mesh(mesh="m2")
    ds.create_device(mesh="m2", device="d2")
    ds.node_model(device="d2", region="r", name="V", equation="x")
    _laplace("d2", "r")

    before = devsim_openblas.calls()
    ds.element_from_edge_model(edge_model="E", device="d2", region="r")

    ex = np.array(ds.get_element_model_values(device="d2", region="r", name="E_x"))
    ey = np.array(ds.get_element_model_values(device="d2", region="r", name="E_y"))
    np.testing.assert_allclose(np.abs(ex), 1.0, rtol=1e-12)
    np.testing.assert_allclose(ey, 0.0, atol=1e-12)
    after = devsim_openblas.calls()
    print("shim calls:", after)
    assert after["dgetrf"] > before["dgetrf"]
    assert after["dgetrs"] > before["dgetrs"]

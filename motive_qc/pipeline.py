"""Pipeline orchestration for Layers 1-5."""

from __future__ import annotations

from motive_qc.artifacts import run_layer4_artifacts
from motive_qc.core import QCValidationError, stop_after_layer
from motive_qc.gaps import run_layer2_gaps
from motive_qc.io import write_outputs
from motive_qc.parse import run_layer1_parse
from motive_qc.report import run_layer5_report
from motive_qc.windows import run_layer3_windows


def run_layers_1_2(config: dict, verbose: bool = False):
    layer1 = run_layer1_parse(config, verbose=verbose)
    if layer1.status == "fail":
        raise QCValidationError("Layer 1 failed validation; Layer 2 was not run.")
    layer2 = run_layer2_gaps(layer1.session, config, verbose=verbose)
    files = write_outputs(layer1, layer2, config, verbose=verbose)
    return layer1, layer2, files


def run_full_pipeline(config: dict, verbose: bool = False):
    layer1 = run_layer1_parse(config, verbose=verbose)
    if layer1.status == "fail":
        raise QCValidationError("Layer 1 failed validation.")
    layer2 = run_layer2_gaps(layer1.session, config, verbose=verbose)

    stop = stop_after_layer(config)
    layer3 = layer4 = layer5 = None

    if stop >= 4:
        layer4 = run_layer4_artifacts(layer1.session, layer2, config, verbose=verbose)
    if stop >= 3:
        layer3 = run_layer3_windows(layer1.session, layer2, layer4, config, verbose=verbose)
    if stop >= 5:
        layer5 = run_layer5_report(layer1, layer2, layer3, layer4, config, verbose=verbose)

    files = write_outputs(
        layer1,
        layer2,
        config,
        verbose=verbose,
        layer3_result=layer3,
        layer4_result=layer4,
        layer5_result=layer5,
    )
    return layer1, layer2, layer3, layer4, layer5, files

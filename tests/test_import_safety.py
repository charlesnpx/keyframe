from types import ModuleType

import pytest

from keyframe.import_safety import defer_optional_pyav_import


def test_defer_optional_pyav_import_blocks_absent_module_on_macos():
    modules = {}

    with defer_optional_pyav_import(system="Darwin", modules=modules):
        assert modules == {"av": None}

    assert modules == {}


def test_defer_optional_pyav_import_preserves_loaded_module():
    av_module = ModuleType("av")
    modules = {"av": av_module}

    with defer_optional_pyav_import(system="Darwin", modules=modules):
        assert modules["av"] is av_module

    assert modules == {"av": av_module}


def test_defer_optional_pyav_import_preserves_module_loaded_inside_guard():
    av_module = ModuleType("av")
    modules = {}

    with defer_optional_pyav_import(system="Darwin", modules=modules):
        modules["av"] = av_module

    assert modules == {"av": av_module}


def test_defer_optional_pyav_import_is_noop_off_macos():
    modules = {}

    with defer_optional_pyav_import(system="Linux", modules=modules):
        assert modules == {}

    assert modules == {}


def test_defer_optional_pyav_import_cleans_up_after_failure():
    modules = {}

    with pytest.raises(RuntimeError, match="failed import"):
        with defer_optional_pyav_import(system="Darwin", modules=modules):
            raise RuntimeError("failed import")

    assert modules == {}

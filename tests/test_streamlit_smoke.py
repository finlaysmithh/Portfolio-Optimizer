from importlib.machinery import SourceFileLoader
from importlib.util import spec_from_loader, module_from_spec
from pathlib import Path


def test_streamlit_projections_import():
    # Ensure the Projections page module loads without raising via path
    path = Path("streamlit_app/pages/3_📈_Projections.py")
    if not path.exists():
        return
    loader = SourceFileLoader("projections_page", str(path))
    spec = spec_from_loader(loader.name, loader)
    mod = module_from_spec(spec)
    loader.exec_module(mod)  # type: ignore[arg-type]

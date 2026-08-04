import asyncio
import tomllib
import logging
import warnings
from pathlib import Path

import pytest
import pytest_asyncio

from simstack.core.context import context
from simstack.core.definitions import DBType
from simstack.tables.model_table import make_model_table
from simstack.tables.node_table import make_node_table

# Suppress pymongo logs/warnings
logging.getLogger("pymongo").setLevel(logging.WARNING)
# Suppress motor logs
logging.getLogger("motor").setLevel(logging.WARNING)

# Suppress Pydantic deprecation warnings
warnings.filterwarnings("ignore", message=".*json_encoders.*")
# Suppress specific pymongo deprecation warnings if any
try:
    from pymongo.errors import PyMongoDeprecationWarning
    warnings.filterwarnings("ignore", category=PyMongoDeprecationWarning)
except ImportError:
    pass

@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest_asyncio.fixture(scope="session", autouse=True)
async def initialized_context():
    """
    Session-based async fixture that initializes the simstack context
    based on variables from simstack_test.toml.
    """
    test_toml_path = Path(__file__).parent / "simstack_test.toml"
    
    with open(test_toml_path, "rb") as f:
        config_data = tomllib.load(f)
    
    parameters = config_data.get("parameters", {})
    general = parameters.get("general", {})
    db_config = parameters.get("db", {})
    
    workdir = Path(general.get("workdir_self", "test_workdir"))
    connection_string = db_config.get("connection_string", "none")
    if connection_string.lower() == "none":
        pytest.skip("Skipping test because connection_string is 'none'")
    db_name = db_config.get("test_database", "test_database")
    
    # Ensure workdir exists
    workdir.mkdir(parents=True, exist_ok=True)
    project_root = Path(__file__).parents[2]
    
    await context.initialize(
        console=False,
        skip_config=True,
        is_test=True,
        resource="self",
        connection_string=connection_string,
        db_type=DBType.MONGODB,
        db_name=db_name,
        workdir=workdir,
        project_root=project_root,
        log_level="DEBUG",
        refresh_mappings=False
    )

    # Initialize model and node tables for both real and mock databases
    dirs = [Path(__file__).parents[1]]

    await make_model_table(context.db, dirs=dirs, drops="src", clear=True,
                           project_root=project_root, ignore_entrypoints=True)
    await make_node_table(context.db, dirs=dirs, drops="src", clear=True,
                          project_root=project_root, ignore_entrypoints=True)

    await context.refresh_mappings()

    yield context
    
    # Cleanup
    if context.initialized:
        if hasattr(context, "db") and context.db:
            await context.db.close()
            context.db = None
        
        context._initialized = False
        context.model_mappings = None
        context.node_mappings = None
        context.resource_config = None
        context.config = None

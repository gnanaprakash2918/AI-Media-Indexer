import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from a2a.types import Message, TextPart, Part, Role, MessageSendParams

from core.agent.a2a_server import check_ollama_connection, create_app
from core.agent.card import get_agent_card
from core.agent.handler import MediaAgentHandler
from core.agent.server import mcp, search_media, ingest_media
from core.schemas import SearchResponse, IngestResponse

# --- A2A Server Tests ---

def test_check_ollama_connection_success():
    """Test successful Ollama connection."""
    with patch("ollama.list") as mock_list:
        check_ollama_connection("llama3")
        mock_list.assert_called_once()

def test_check_ollama_connection_failure():
    """Test Ollama connection failure."""
    with patch("ollama.list", side_effect=Exception("Connection refused")):
        with pytest.raises(RuntimeError, match="Ollama is not running"):
            check_ollama_connection("llama3")

@patch("core.agent.a2a_server.check_ollama_connection")
@patch("core.agent.a2a_server.get_agent_card")
@patch("core.agent.a2a_server.MediaAgentHandler")
def test_create_app(mock_handler, mock_card, mock_check):
    """Test FastAPI app creation."""
    app = create_app()
    assert app is not None
    mock_check.assert_called_once()

# --- Card Tests ---

def test_get_agent_card_structure():
    card = get_agent_card("http://test-host:1234")

    assert card.name == "MediaIndexer"
    assert card.version == "1.0.0"
    assert "test-host:1234/a2a" in card.url

    # Verify skills
    skill_ids = [s.id for s in card.skills]
    assert "search_media" in skill_ids
    assert "ingest_media" in skill_ids

    # Verify capabilities
    assert card.capabilities.streaming is True

# --- Handler Tests ---

@pytest.fixture
def agent_handler():
    return MediaAgentHandler(model_name="test-model")

@pytest.mark.asyncio
async def test_handler_initialization(agent_handler):
    assert agent_handler.model_name == "test-model"
    assert len(agent_handler.tools) == 2  # search/ingest schemas

@pytest.mark.asyncio
async def test_on_message_send_empty(agent_handler):
    # If no text, should return generic reply
    msg = Message(role=Role.user, message_id="1", context_id="c1", parts=[])
    params = MessageSendParams(message=msg)

    task = await agent_handler.on_message_send(params)
    reply = task.status.message.parts[0].root.text
    assert "didn't receive any text" in reply

@pytest.mark.asyncio
async def test_on_message_send_ollama_call(agent_handler):
    # Simulate user text
    txt_part = Part(root=TextPart(text="find cat"))
    msg = Message(role=Role.user, message_id="1", context_id="c1", parts=[txt_part])
    params = MessageSendParams(message=msg)

    # Mock Ollama response (no tools)
    mock_ollama_resp = {
        "message": {"role": "assistant", "content": "I found a cat"}
    }

    with patch("ollama.chat", return_value=mock_ollama_resp) as mock_chat:
        task = await agent_handler.on_message_send(params)

        # Verify Ollama called with correct model
        mock_chat.assert_called()
        args, kwargs = mock_chat.call_args
        assert kwargs["model"] == "test-model"

        # Verify reply
        reply = task.status.message.parts[0].root.text
        assert reply == "I found a cat"

@pytest.mark.asyncio
async def test_execute_tool_search(agent_handler):
    # Mock the internal logic of _execute_tool
    args = {"query": "test", "limit": 1}

    # We mock search_media imported in handler
    with patch("core.agent.handler.search_media", new_callable=AsyncMock) as mock_search:
        mock_result = MagicMock()
        mock_result.model_dump.return_value = {
            "visual_matches": [{"time": "1s", "file": "v.mp4", "content": "c"}],
            "dialogue_matches": []
        }
        mock_search.return_value = mock_result

        summary = await agent_handler._execute_tool("search_media", args)

        assert "Search results for: 'test'" in summary
        assert "[V] 1s v.mp4: c" in summary

# --- MCP Server Tests ---

@pytest.mark.asyncio
async def test_mcp_list_tools():
    """Test that MCP tools are correctly registered."""
    tools = await mcp.list_tools()
    tool_names = [t.name for t in tools]
    assert "search_media" in tool_names
    assert "ingest_media" in tool_names

@pytest.mark.asyncio
@patch("core.agent.server._get_search_engine")
async def test_search_media_tool(mock_get_engine):
    """Test the search_media tool wrapper."""
    mock_engine = MagicMock()
    mock_engine.search.return_value = {
        "visual_matches": [],
        "dialogue_matches": []
    }
    mock_get_engine.return_value = mock_engine

    result = await search_media(query="test query")

    assert isinstance(result, SearchResponse)
    mock_engine.search.assert_called_once_with("test query", limit=5)

@pytest.mark.asyncio
@patch("core.agent.server._get_pipeline")
@patch("pathlib.Path.exists")
@patch("pathlib.Path.is_file")
async def test_ingest_media_tool(mock_is_file, mock_exists, mock_get_pipeline):
    """Test the ingest_media tool wrapper."""
    mock_exists.return_value = True
    mock_is_file.return_value = True

    mock_pipeline = AsyncMock()
    mock_get_pipeline.return_value = mock_pipeline

    # Mock full path resolution to avoid OS dependency in test
    with patch("pathlib.Path.expanduser") as mock_expand:
        mock_expand.resolve.return_value = MagicMock()

        result = await ingest_media(file_path="test_video_mp4")

        assert isinstance(result, IngestResponse)
        assert "Ingestion complete" in result.message
        mock_pipeline.process_video.assert_called_once()

@pytest.mark.asyncio
@patch("pathlib.Path.exists")
async def test_ingest_media_file_not_found(mock_exists):
    """Test ingestion with non-existent file."""
    mock_exists.return_value = False

    result = await ingest_media(file_path="fake.mp4")

    assert "Error: File not found" in result.message

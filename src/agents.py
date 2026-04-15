import base64
from typing import Annotated
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


# =============================================================================
# AGENT STATE
# =============================================================================

class AgentState(TypedDict):
    """The state passed between nodes in the LangGraph agent graph.

    messages: Full conversation history. add_messages appends new messages
              rather than replacing the list on each graph step.
    """
    messages: Annotated[list, add_messages]


# =============================================================================
# AGENT
# =============================================================================

class Agent:
    """
    A LangGraph-powered LLM agent supporting text prompts, image prompts,
    and tool use across multiple providers (OpenAI, Gemini, Groq, HuggingFace).

    When tools are supplied the agent runs a full ReAct-style loop:
    the LLM decides whether to call a tool, the ToolNode executes it,
    and the result is fed back to the LLM until it produces a final answer.

    When no tools are supplied the agent runs a single-step LLM call,
    making it a lightweight drop-in for simple use cases.

    Supports both blocking (run) and streaming (stream) execution modes
    through a shared message-building pipeline.

    Args:
        llm_provider (str): Provider name — "OpenAI", "Gemini", "Groq", "HuggingFace".
        model_name (str): Model identifier passed to the provider.
        temperature (float): Sampling temperature (0.0 = deterministic).
        api_key (str, optional): API key for the provider.
        tools (list, optional): LangChain @tool functions available to the agent.
        system_prompt (str, optional): System-level instruction prepended to every call.
        memory (bool): If True, enables multi-turn conversation memory via MemorySaver.
                       Each thread_id maintains its own isolated conversation history.
    """

    def __init__(
        self,
        llm_provider: str = "OpenAI",
        model_name: str = "gpt-4o-mini",
        temperature: float = 0.0,
        api_key: str = None,
        tools: list = None,
        system_prompt: str = None,
    ):
        self.llm_provider = llm_provider
        self.model_name   = model_name
        self.temperature  = temperature
        self.api_key      = api_key
        self.tools        = tools or []
        self.system_prompt = system_prompt

        # Initialise the base LLM for the configured provider
        self.llm = self._create_llm()

        # If tools are provided, bind them to the LLM so it knows they exist
        # and can generate tool-call requests in its responses
        if self.tools:
            self.llm = self.llm.bind_tools(self.tools)


    # -------------------------------------------------------------------------
    # LLM SETUP
    # -------------------------------------------------------------------------

    def _create_llm(self) -> BaseChatModel:
        """Instantiate the LangChain chat model for the configured provider.

        API keys are passed explicitly rather than relying on environment
        variable naming conventions which differ per provider.

        Returns:
            A LangChain BaseChatModel instance.

        Raises:
            ValueError: If the provider name is not recognised.
            ImportError: If the required integration package is not installed.
        """
        if self.llm_provider == "OpenAI":
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(
                model=self.model_name,
                temperature=self.temperature,
                api_key=self.api_key,
            )

        elif self.llm_provider == "Google":
            from langchain_google_genai import ChatGoogleGenerativeAI
            return ChatGoogleGenerativeAI(
                model=self.model_name,
                temperature=self.temperature,
                google_api_key=self.api_key,
            )

        elif self.llm_provider == "Groq":
            from langchain_groq import ChatGroq
            return ChatGroq(
                model=self.model_name,
                temperature=self.temperature,
                api_key=self.api_key,
            )

        elif self.llm_provider == "HuggingFace":
            from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
            endpoint = HuggingFaceEndpoint(
                repo_id=self.model_name,
                temperature=self.temperature,
                huggingfacehub_api_token=self.api_key,
            )
            return ChatHuggingFace(llm=endpoint)

        else:
            raise ValueError(f"Unsupported LLM provider: {self.llm_provider}")


    def chatbot_node(self, state: AgentState) -> dict:
        """LangGraph node that invokes the LLM with the current message history.

        Prepends the system prompt on every call so it is always in context
        regardless of how many tool rounds have occurred.

        Args:
            state: The current agent state containing the message history.

        Returns:
            A dict with the LLM response appended to the message list.
        """
        messages = state["messages"]

        # ONLY prepend the system prompt if it's not already the first message
        # This prevents 'System Message' fatigue and formatting errors
        if self.system_prompt:
            if not messages or not isinstance(messages[0], SystemMessage):
                messages = [SystemMessage(content=self.system_prompt)] + messages

        response = self.llm.invoke(messages)

        return {"messages": [response]}

    # -------------------------------------------------------------------------
    # MESSAGE BUILDERS
    # -------------------------------------------------------------------------

    def _build_image_messages(
        self,
        user_prompt: str,
        image_data: bytes | list[bytes],
        mime_type: str | list[str] = "image/jpeg",
    ) -> list:
        """Build a multimodal HumanMessage containing text and one or more images.

        Normalises single image / single mime_type inputs to lists so the
        same code path handles all cases. Each image is base64-encoded and
        embedded as a data URL — the standard LangChain vision format across
        all supported providers.

        Args:
            user_prompt: Text prompt to send alongside the image(s).
            image_data: Single image bytes or list of image bytes.
            mime_type: Single MIME type string (applied to all images) or a
                       list of MIME types matching each image. Defaults to
                       "image/jpeg".

        Returns:
            A list containing a single multimodal HumanMessage.
        """
        # Normalise to lists so single and multiple images use the same path
        if isinstance(image_data, bytes):
            image_data = [image_data]
        if isinstance(mime_type, str):
            mime_type = [mime_type] * len(image_data)

        # Start content block with the text prompt
        content = [{"type": "text", "text": user_prompt}]

        # Append each image as a separate base64-encoded content block
        for img, mime in zip(image_data, mime_type):
            image_b64 = base64.b64encode(img).decode("utf-8")
            image_url = f"data:{mime};base64,{image_b64}"
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": image_url,
                    "detail": "auto"
                }
            })

        return [HumanMessage(content=content)]

    # -------------------------------------------------------------------------
    # SHARED PIPELINE
    # -------------------------------------------------------------------------

    def prepare(
        self,
        user_prompt: str,
        image_data: bytes | list[bytes] | None,
        mime_type: str | list[str],
    ) -> dict:
        """Build the initial graph state and config shared by run() and stream().

        Centralising this logic means run() and stream() never duplicate
        message building or config construction — they just call _prepare()
        and pass the results to graph.invoke() or graph.stream() respectively.

        Args:
            user_prompt: The user's text prompt.
            image_data: Optional image bytes or list of image bytes.
            mime_type: MIME type(s) for the image(s).
            thread_id: Conversation thread ID for memory isolation.

        Returns:
            A tuple of (initial_state dict, config dict) ready to pass
            directly to graph.invoke() or graph.stream().
        """
        # Build messages with or without images
        if image_data is not None:
            messages = self._build_image_messages(user_prompt, image_data, mime_type)
        else:
            messages = [HumanMessage(content=user_prompt)]


        return {"messages": messages}

        """Build the initial graph state and config shared by run() and stream().

        Centralising this logic means run() and stream() never duplicate
        message building or config construction — they just call _prepare()
        and pass the results to graph.invoke() or graph.stream() respectively.

        Args:
            user_prompt: The user's text prompt.
            image_data: Optional image bytes or list of image bytes.
            mime_type: MIME type(s) for the image(s).
            thread_id: Conversation thread ID for memory isolation.

        Returns:
            A tuple of (initial_state dict, config dict) ready to pass
            directly to graph.invoke() or graph.stream().
        """
        # Build messages with or without images
        if image_data is not None:
            messages = self._build_image_messages(user_prompt, image_data, mime_type)
        else:
            messages = [HumanMessage(content=user_prompt)]


        return {"messages": messages}


import streamlit as st
import uuid
import json
from src.agents import Agent
from src.image_uploader import ImageUploader


# =============================================================================
# HELPERS
# =============================================================================

def load_models(json_path: str) -> dict:
    """
    Parses a JSON configuration file to retrieve AI model metadata.

    Instead of raising exceptions, this function catches them and displays 
    user-friendly error messages in the Streamlit UI before halting execution.

    Args:
        json_path (str): The file path to the models.json configuration file.

    Returns:
        dict: A nested dictionary of model details if successful.
    
    Error Handling:
        - On FileNotFoundError: Displays error and stops the app.
        - On JSONDecodeError: Displays syntax error details and stops the app.
        - On general Exception: Displays the traceback and stops the app.
    """
    try:
        # Use utf-8 to ensure special characters in descriptions load correctly
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)
          
    except FileNotFoundError:
        st.error(f"🚨 **Configuration Missing:** The file '{json_path}' was not found.")
        st.stop() 

    except json.JSONDecodeError as e:
        st.error(f"📑 **Syntax Error:** The file '{json_path}' contains invalid JSON.")
        st.info(f"Check line {e.lineno}, column {e.colno}.")
        st.stop() 

    except Exception as e:
        st.error(f"❌ **Unexpected Error:** An issue occurred while loading '{json_path}'.")
        st.exception(e) # Collapsible technical details for debugging
        st.stop()

def get_models_for_provider(models_data: dict, provider: str) -> dict:
    """
    Extracts the subset of models belonging to a specific AI provider.

    This helper function filters the global models dictionary to return only 
    the models associated with the user's selected provider (e.g., 'Google'). 
    It uses a fallback mechanism to prevent the UI from breaking if a 
    provider key is missing.

    Args:
        models_data (dict): The full dictionary loaded from models.json.
        provider (str): The name of the provider to look up (e.g., "OpenAI").

    Returns:
        dict: A dictionary of models for that provider, or an empty 
              dictionary {} if the provider is not found.
    """
    # Using .get() instead of models_data[provider] is a "defensive" move.
    # If 'provider' isn't a key in the dictionary, it returns the second 
    # argument ({}) instead of raising a KeyError.
    return models_data.get(provider, {})

def render_model_description(info: dict, provider: str) -> None:
    """
    Renders a formatted UI block showing model metadata and pricing.

    This function takes a dictionary of model information and displays it 
    using a mix of Streamlit's native components and raw HTML/CSS. It 
    provides the user with the model's originating company, a brief 
    functional description, and the cost per million tokens.

    Args:
        info (dict): The specific model dictionary from models.json 
                     (e.g., {"Company": "Google", "Input": 3.50, ...}).
        provider (str): The fallback provider name if 'Company' is missing.

    Returns:
        None
    """

    # 1. DATA EXTRACTION: Pull the pricing or default to "N/A" to prevent errors.
    input_cost  = info.get("Input",  "N/A")
    output_cost = info.get("Output", "N/A")

    # 2. HEADER: Displays the building emoji and the company name in bold.
    st.caption(f"🏢 **{info.get('Company', provider)}**")

    # 3. BODY: Displays the model's intended use case or description.
    st.write(info.get("Description", "No description available."))

    # 4. FOOTER (HTML): Creates a small, grey, "pro-style" pricing line.
    # Uses HTML entities: &#128176; (Money Bag) and &#36; (Dollar Sign).
    st.markdown(
        f"<p style='color:grey;font-size:0.85em;margin:0'>"
        f"&#128176; Input: &#36;{input_cost} / 1M tokens"
        f" &nbsp;&middot;&nbsp; "
        f"Output: &#36;{output_cost} / 1M tokens"
        f"</p>",
        unsafe_allow_html=True,
    )

def load_css(css_path: str) -> None:
    """Inject a local CSS stylesheet into the Streamlit app."""
    with open(css_path, "r") as f:
        css = f.read()
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def on_image_delete() -> None:
    """Callback triggered when the user deletes all uploaded images.

    Does not clear chat history — the conversation continues without images.
    Only clears the image hash so the uploader resets cleanly.
    """
    st.session_state.last_image_hash = None


def build_context_prefix(has_images: bool) -> str:
    """Build an optional context prefix to prepend to the user's message.

    Informs the LLM of any images provided so it can
    factor them into its response without the user repeating this context
    in every message. The prefix is sent to the LLM but not shown in the
    chat history — the user only sees their clean question.

    Args:
        has_images: Whether images are currently uploaded.

    Returns:
        A context prefix string, or empty string if no context exists.
    """
    parts = []

    if has_images:
        parts.append("[image(s) attached — please reference them in your response]")

    if parts:
        # Separate context from the user's actual question with a blank line
        return "\n".join(parts) + "\n\n"

    return ""

def model_selector(slot: str, models_data: dict, exclude_model: str | None) -> tuple:
    """
    Renders a Streamlit UI component for selecting an LLM provider and model.

    This function handles the state lifecycle for a specific model 'slot'. If the model 
    selection changes, it clears the chat history, regenerates session IDs, and 
    destroys the existing agent instance to ensure the new model starts fresh.

    Args:
        slot (str): Unique identifier for the UI slot (e.g., 'left', 'right', 'a').
        models_data (dict): Nested dictionary containing provider and model metadata.
        exclude_model (str | None): A model name to filter out (used to prevent 
            selecting the same model in two different slots).

    Returns:
        tuple: (provider_name, model_name) or (provider_name, None) if no model selected.
    """

    with st.container(border=True):
        # 1. Initialize Provider Selection
        if slot == "1":
            provider = st.selectbox(
                "Primary Provider", 
                PROVIDERS, 
                key=f"provider_{slot}"
            )
        else:
            provider = st.selectbox(
                "Secondary Provider", 
                PROVIDERS, 
                key=f"provider_{slot}"
            )

        # 2. Fetch models for the chosen provider and filter exclusions
        provider_models = get_models_for_provider(models_data, provider)
        all_names = list(provider_models.keys())

        options = ["Select Model..."] + [m for m in all_names if m != exclude_model]

        # 3. Render Model Selection
        if slot == "1":
            model = st.selectbox(
                "Primary Model", 
                options, 
                key=f"model_{slot}"
            )
        else:
            model = st.selectbox(
                "Secondary Model", 
                options, 
                key=f"model_{slot}"
            )

        # 4. State Management Logic: Detect model changes
        tracker_key = f"model_tracker_{slot}"
        last_val = st.session_state.get(tracker_key)

        if model != last_val:
            # A. Wipe the history for this specific slot
            st.session_state[f"messages_{slot}"] = []
            
            # B. Generate a fresh thread ID for the new model
            st.session_state[f"session_id_{slot}"] = str(uuid.uuid4())
            
            # C. KILL the agent so it rebuilds with the new model name/key
            if f"agent_{slot}" in st.session_state:
                del st.session_state[f"agent_{slot}"]
            
            # D. UPDATE THE TRACKER (Crucial: sync the tracker to the new model)
            st.session_state[tracker_key] = model
            
            # E. RERUN to refresh the UI and clear the chat boxes
            st.rerun()

        # 5. UI Feedback: Show model details if a valid selection is made
        if model and model != "Select Model...":
            # Pull metadata from the dictionary subset
            model_info = provider_models.get(model, {})
            render_model_description(model_info, provider)

            return provider, model

    return provider, None

def clear_chat_callback(slot: str):
    """
    Performs a targeted reset of the conversation state for a specific UI slot.

    This callback ensures that only the data associated with the provided 'slot' 
    identifier is purged, allowing other slots to remain unaffected. It handles 
    message history deletion, session ID rotation, and agent instance destruction.

    Args:
        slot (str): The unique identifier (e.g., 'left', 'right') representing 
                   the conversation stream to be cleared.
    """

    # 1. Clear messages for THIS slot only
    st.session_state[f"messages_{slot}"] = []
    
    # 2. Reset the thread ID for a fresh start
    st.session_state[f"session_id_{slot}"] = str(uuid.uuid4())
    
    # 3. Kill the agent so it rebuilds
    if f"agent_{slot}" in st.session_state:
        del st.session_state[f"agent_{slot}"]


def api_key_handler(slot: str, provider: str):
    """
    Manages secure API key entry and persistence for a specific model slot.

    This function implements a state-locked 'vault' mechanism. It provides a 
    toggle between a masked input state and an 'Active' state. When a key is 
    updated, it invalidates any existing agents to ensure credentials are 
    refreshed in the backend.

    Args:
        slot (str): Unique identifier for the UI slot ('left', 'right', etc.).
        provider (str): Name of the LLM provider (e.g., 'OpenAI', 'Anthropic').

    Returns:
        str | None: The active API key from the session vault, or None if empty.
    """

    vault_key = f"confirmed_api_key_{slot}"
    widget_key = f"widget_input_{slot}"
    error_key = f"api_key_error_{slot}"

    # 1. THE PERSISTENCE CHECK
    # If a key is in the vault, show the Success message.
    if st.session_state.get(vault_key):
        st.success(f"✅ {provider} API Key Active")
        if st.button(f"Update {provider} Key", key=f"btn_update_{slot}"):
            st.session_state[vault_key] = "" # Clear vault
            if f"agent_{slot}" in st.session_state:
                del st.session_state[f"agent_{slot}"]
            st.rerun()
        return st.session_state[vault_key]

    # 2. THE INPUT UI: Rendered only if no valid key exists in the vault
    st.markdown(f"**Enter {provider} API Key**")
    
    api_key_input = st.text_input(
        label="API Key Input",
        type="password",
        placeholder="Paste key here...",
        value=st.session_state.get(vault_key, ""), # Pull from vault
        key=widget_key,
        label_visibility="collapsed",
    )

    # 3. VALIDATION & SAVE LOGIC
    if st.button("Enter", use_container_width=True, key=f"btn_save_key_{slot}"):
        if not api_key_input.strip():
            st.session_state[error_key] = "⚠️ API key cannot be blank."
        else:
            # Commit the key to the vault and clear any previous errors
            st.session_state[vault_key] = api_key_input.strip()
            st.session_state[error_key] = "" 

            # Kill the agent instance to force a fresh connection with the new key
            if f"agent_{slot}" in st.session_state:
                del st.session_state[f"agent_{slot}"]

        # Rerun to switch from the Input UI to the Success UI
        st.rerun()

    # 4. ERROR DISPLAY: Persistent error messaging tied to the slot
    if st.session_state.get(error_key):
        st.error(st.session_state[error_key])

    return st.session_state.get(vault_key)

def stream_response(
    agent: Agent,
    user_message: str,
    thread_id: str,
    image_data: list[bytes] | None = None,
    mime_type: str | list[str] = "image/jpeg",
) -> str:
    """Stream the agent response token by token into a Streamlit chat bubble.

    Displays a live-updating response as the model generates tokens,
    then returns the full accumulated string for saving to chat history.

    Args:
        agent: The LLM agent to query.
        user_message: The full message to send (context prefix + user question).
        thread_id: The conversation thread ID for memory continuity.
        image_data: Optional list of image bytes to include in the message.
                    If None, the call is text-only (images already in memory).

    Returns:
        The complete response text after streaming finishes.
    """
    full_response = ""

    # Generator that accumulates chunks as they arrive for saving to history
    def response_generator():
        nonlocal full_response
        for chunk in agent.stream(
            user_prompt=user_message,
            image_data=image_data,
            mime_type=mime_type,
            thread_id=thread_id,
        ):
            full_response += chunk
            yield chunk

    # Render streaming response inside an assistant chat bubble
    with st.chat_message("assistant"):
        st.write_stream(response_generator())

    return full_response


# =============================================================================
# INITIALISATION
# =============================================================================

def init_session_state() -> None:
    """
    Initializes and synchronizes all session-state variables on the first run.

    This function acts as the state manager for the application. It ensures that 
    all necessary keys exist in st.session_state before the UI renders. It also 
    handles the re-attachment of non-serializable objects (like callbacks) that 
    Streamlit drops between execution cycles.

    Note:
        Must be called at the very top of the script's entry point.
    """

    # 1. Component State Persistence
    # Initializing the custom ImageUploader class. We check for existence so we 
    # don't overwrite existing uploads during a standard rerun.
    if "uploader" not in st.session_state:
        st.session_state.uploader = ImageUploader(on_delete=on_image_delete)

    # 2. Callback Re-synchronization
    # Re-attach callback every rerun — Streamlit does not serialise callables
    st.session_state.uploader.on_delete = on_image_delete

    # 3. Multi-Slot Architecture Initialization
    # Using a loop to standardize the setup for dual-agent comparison (Slots 1 and 2).
    for slot in ["1", "2"]:
        # Stores the list of chat dictionaries {role, content}
        if f"messages_{slot}" not in st.session_state:
            st.session_state[f"messages_{slot}"] = []
        
        # Unique identifier for the conversation thread (useful for LangChain/Agent logging)
        if f"session_id_{slot}" not in st.session_state:
            st.session_state[f"session_id_{slot}"] = str(uuid.uuid4())

        # Monitors if the user changed the model in the dropdown    
        if f"model_tracker_{slot}" not in st.session_state:
            st.session_state[f"model_tracker_{slot}"] = "Select Model..."

        # The 'Vault' key for API credentials
        if f"confirmed_api_key_{slot}" not in st.session_state:
            st.session_state[f"confirmed_api_key_{slot}"] = ""

        # Tracks the previous provider to detect provider-level changes
        if f"last_provider_{slot}" not in st.session_state:
            st.session_state[f"last_provider_{slot}"] = None

    # 4. Global Vision Logic
    # Used to detect if the set of uploaded images has changed since the last LLM call
    if "last_image_hash" not in st.session_state:
        st.session_state.last_image_hash = None

    # Holds the temporary stream output from the LLM before it's committed to history
    if "current_full_message" not in st.session_state:
        st.session_state.current_full_message = ""

    # Stores the processed base64 strings or file objects currently staged for the LLM
    if "current_images_to_send" not in st.session_state:
        st.session_state.current_images_to_send = None


# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="🤖 Multi-Model Vision Hub",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# 1. Setup Constants
PROVIDERS = ["OpenAI", "Google", "Groq", "HuggingFace"]

SYSTEM_PROMPT = """You are an expert in image analysis.
Provide clear, concise, accurate responses using bullet points and examples when helpful.
Respond in friendly, conversational and professional tone."""

# Stylesheet now handles all styling including the diagnosis text area —
# no inline CSS needed in this file.
load_css("static/style.css")
init_session_state()

# 3. Load Model Data (JSON)
models_data = load_models("models.json")

st.markdown(f"""
    <div style="display: flex; align-items: center;">
        <h1 style="margin: 0;">🤖 Multi-Model Vision Hub</h1>
    </div>
""", unsafe_allow_html=True)


col1, col2, col3 = st.columns([1, 1.2, 1.2])


# =============================================================================
# UI — INPUT PANEL (diagnosis + images)
# =============================================================================

# Visually distinct input panel above the chat area.
# Styling (border, background) is handled by style.css.
with col1:
    # Aesthetically centered instruction for the user
    st.markdown(
    "<p style='text-align: center; color: #aaaaaa;'>Upload an image, and place your inquiries below.</p>",
    unsafe_allow_html=True
    )

    # Render the custom uploader component (handles internal state/preview)
    st.session_state.uploader.render()
    img_data = st.session_state.uploader.get_image()

    # Capture user input from the global chat bar
    query = st.chat_input("Ask a question about the image...")
    if query:

        # 1. DATA GATHERING & STATE ANALYSIS
        # Extract binary data and a unique hash representing the current set of images
        image_bytes    = st.session_state.uploader.get_images()
        image_hash     = st.session_state.uploader.get_hash()

        # Get the mime types from your uploader component
        # If your component doesn't have get_types(), you might need to add it 
        # to return a list like ["image/png", "image/jpeg"]
        image_types = st.session_state.uploader.get_types()

        # Detect if the images are new or if this is a follow-up to the existing set
        has_images     = len(image_bytes) > 0
        images_changed = image_hash != st.session_state.last_image_hash

        # 2. CONTEXT PREPARATION
        # Build the final prompt string (prepends context like "Given the image...")
        st.session_state.current_full_message = build_context_prefix(has_images) + query

        # Broadcast the message to both conversation histories for side-by-side comparison
        st.session_state.messages_1.append({"role": "user", "content": st.session_state.current_full_message})
        st.session_state.messages_2.append({"role": "user", "content": st.session_state.current_full_message})

        # 3. INTELLIGENT IMAGE SENDING LOGIC
        # We only stage images for the LLM if they are present and have changed.
        if has_images and images_changed:
            st.session_state.current_images_to_send = image_bytes
            # Store the types in session state so the agent can use them
            st.session_state.current_mime_types = image_types

            # Update the persistent hash to "lock" this set of images until they are modified
            st.session_state.last_image_hash = image_hash
        else:
            # For follow-up questions, we send None (LLM uses its memory)
            st.session_state.current_images_to_send = None
            st.session_state.current_mime_types = "image/jpeg" # Default fallback



# =============================================================================
# UI — CHAT AREA
# =============================================================================

with col2:
    st.header("Primary AI Model")

    # 1. MODEL #1 MODEL & CREDENTIAL INITIALIZATION
    # Renders selectors and ensures this slot doesn't overlap with the other side's model
    prov_1, mod_1 = model_selector("1", models_data, exclude_model=st.session_state.get("model_2"))

    if mod_1 and mod_1 != "Select Model...":
        key_1 = api_key_handler("1", prov_1)

        # Lazy initialization of the Agent: only builds if a key exists and agent is missing
        if key_1 and "agent_1" not in st.session_state:
            st.session_state.agent_1 = Agent(
                llm_provider=prov_1,
                model_name=mod_1,
                temperature=0.0,
                api_key=key_1,
                system_prompt=SYSTEM_PROMPT,
                memory=True,
                )

    # 2. CHAT INTERFACE CONTAINER
    with st.container(height=500, border=True):
        # --- Render existing chat history ---
        for msg in st.session_state.messages_1:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        # 3. TRIGGER INFERENCE (Model Response Logic)
        # We only trigger a response if the last message in history is from the 'user'
        if st.session_state.messages_1 and st.session_state.messages_1[-1]["role"] == "user":
            if "agent_1" in st.session_state:
                # Calls the streaming helper to display the AI's "thought" process
                res = stream_response(
                    agent=st.session_state.agent_1,
                    user_message=st.session_state.current_full_message, # Use persisted message
                    image_data=st.session_state.current_images_to_send, # Use persisted images
                    mime_type=st.session_state.get("current_mime_types", "image/jpeg"),
                    thread_id=st.session_state.session_id_1
                )

                # Commit the response to history and flag a rerun to update the UI
                st.session_state.messages_1.append({"role": "assistant", "content": res})
                st.session_state.needs_rerun = True

    # 4. SLOT-SPECIFIC UTILITIES
    if st.session_state.messages_1:
        st.button("🗑️ Clear Primary AI Model Chat", 
                  key="btn_clear_1", 
                  use_container_width=True, 
                  on_click=clear_chat_callback, 
                  args=("1",))
    

with col3:
    st.header("Secondary AI Model")

    # 1. MODEL #2 MODEL & CREDENTIAL INITIALIZATION
    # Renders selectors and ensures this slot doesn't overlap with the other side's model
    prov_2, mod_2 =model_selector("2", models_data, exclude_model=st.session_state.get("model_1"))

    if mod_2 and mod_2 != "Select Model...":
        key_2 = api_key_handler("2", prov_2)

        # Lazy initialization of the Agent: only builds if a key exists and agent is missing
        if key_2 and "agent_2" not in st.session_state:
            st.session_state.agent_2 = Agent(
                llm_provider=prov_2,
                model_name=mod_2,
                temperature=0.0,
                api_key=key_2,
                system_prompt=SYSTEM_PROMPT,
                memory=True,
                )

    # 2. CHAT INTERFACE CONTAINER
    with st.container(height=500, border=True):
        for msg in st.session_state.messages_2:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        # 3. TRIGGER INFERENCE (Model Response Logic)
        if st.session_state.messages_2 and st.session_state.messages_2[-1]["role"] == "user":
            if "agent_2" in st.session_state:
                # Calls the streaming helper to display the AI's "thought" process
                res = stream_response(
                    agent=st.session_state.agent_2,
                    user_message=st.session_state.current_full_message, # Use persisted message
                    image_data=st.session_state.current_images_to_send, # Use persisted images
                    mime_type=st.session_state.get("current_mime_types", "image/jpeg"),
                    thread_id=st.session_state.session_id_2
                )

                # Commit the response to history and flag a rerun to update the UI
                st.session_state.messages_2.append({"role": "assistant", "content": res})
                st.session_state.needs_rerun = True

    # 4. SLOT-SPECIFIC UTILITIES
    if st.session_state.messages_2:
        st.button("🗑️ Clear Secondary AI Model Chat", 
                  key="btn_clear_2", 
                  use_container_width=True, 
                  on_click=clear_chat_callback, 
                  args=("2",))

# GLOBAL RERUN HANDLER
# This block is essential for the 'Streamlit Flow'. Because we appended messages 
# to the state AFTER the containers were drawn, we must rerun once to make 
# those assistant messages visible to the user immediately.
if st.session_state.get("needs_rerun"):
    st.session_state.needs_rerun = False
    st.rerun()
import os
import io
import hashlib
import imghdr
import streamlit as st


class ImageUploader:
    """
    Handles uploading, validation, display, deletion and hash tracking
    for one or more images in a Streamlit app.

    Extends the original single-image design to support multiple images
    while preserving all validation, hashing, preview and delete behaviour.

    Uses Streamlit session state to persist image data across reruns.
    A combined MD5 hash of all uploaded images is used to detect changes
    and prevent redundant LLM calls.
    """

    # Allowed MIME types for uploaded files
    ALLOWED_TYPES = ["image/jpeg", "image/png", "image/jpg"]

    # Allowed file extensions for uploaded files
    ALLOWED_EXTENSIONS = [".jpg", ".jpeg", ".png"]

    def __init__(self, on_delete=None):
        # Optional callback to run when images are deleted —
        # lets the caller clean up its own state (e.g. clear LLM response)
        self.on_delete = on_delete

        # Initialise session state variables on first instantiation
        self._init_session_state()

        if "uploaded_types" not in st.session_state:
            st.session_state.uploaded_types = [] # Store MIME types here

    # -------------------------------------------------------------------------
    # SESSION STATE
    # -------------------------------------------------------------------------

    def _init_session_state(self):
        """Initialise all session state variables on the first run.

        Uses 'not in' checks so existing values are never overwritten
        on subsequent Streamlit reruns.
        """
        if "uploaded_files" not in st.session_state:
            # List of raw bytes, one entry per uploaded image
            st.session_state.uploaded_files = []

        if "uploaded_names" not in st.session_state:
            # List of original filenames, parallel to uploaded_files
            st.session_state.uploaded_names = []

        if "upload_key" not in st.session_state:
            # Incrementing this integer forces Streamlit to render a fresh
            # uploader widget, effectively resetting it
            st.session_state.upload_key = 0

        if "image_hash" not in st.session_state:
            # Combined MD5 hash of all current image bytes.
            # Used to detect when the set of images has changed.
            st.session_state.image_hash = None

    # -------------------------------------------------------------------------
    # VALIDATION
    # -------------------------------------------------------------------------

    def _validate(self, uploaded_file) -> bool:
        """Validate a single uploaded file using three independent checks.

        Checks are ordered from cheapest to most expensive:
          1. MIME type reported by Streamlit — catches obvious wrong types
          2. File extension — catches files with wrong or missing extensions
          3. Actual file content via imghdr — cannot be fooled by renaming

        Args:
            uploaded_file: A single file object from st.file_uploader.

        Returns:
            True if all checks pass, False if any check fails.
        """
        # Check 1 — MIME type reported by Streamlit
        if uploaded_file.type not in self.ALLOWED_TYPES:
            st.error(
                f"❌ {uploaded_file.name}: Invalid file type '{uploaded_file.type}'. "
                f"Please upload a JPG or PNG image."
            )
            return False

        # Check 2 — file extension
        ext = os.path.splitext(uploaded_file.name)[1].lower()
        if ext not in self.ALLOWED_EXTENSIONS:
            st.error(
                f"❌ {uploaded_file.name}: Invalid extension '{ext}'. "
                f"Please upload a JPG or PNG image."
            )
            return False

        # Check 3 — actual file content using imghdr
        # Reads raw bytes to confirm the file really is an image,
        # regardless of what the MIME type or extension claim
        file_bytes = uploaded_file.read()
        actual_type = imghdr.what(io.BytesIO(file_bytes))
        if actual_type not in ("jpeg", "png"):
            st.error(
                f"❌ {uploaded_file.name}: File content does not appear "
                f"to be a valid image."
            )
            return False

        return True

    # -------------------------------------------------------------------------
    # SESSION STATE MANAGEMENT
    # -------------------------------------------------------------------------

    def _save_to_session(self, uploaded_files: list):
        """Read, validate and save all uploaded files to session state.

        Only files that pass all three validation checks are saved.
        The file pointer is reset to the start before reading bytes so
        that data is not lost after the validation read.

        Args:
            uploaded_files: List of file objects from st.file_uploader.
        """
        valid_bytes = []
        valid_names = []
        valid_types = []

        for f in uploaded_files:
            if self._validate(f):
                # Reset pointer after validation read so we can read bytes again
                f.seek(0)
                valid_bytes.append(f.read())
                valid_names.append(f.name)
                valid_types.append(f.type)

        # Replace session state with the validated set
        st.session_state.uploaded_files = valid_bytes
        st.session_state.uploaded_names = valid_names
        st.session_state.uploaded_types = valid_types

        # Compute a combined hash across all images so any change is detected
        st.session_state.image_hash = self._compute_hash(valid_bytes)

    def _delete(self):
        """Clear all image-related session state and reset the uploader widget.

        Incrementing upload_key forces Streamlit to render a completely fresh
        uploader widget, clearing any files the user had selected.
        Triggers a rerun so the UI updates immediately.
        """
        st.session_state.uploaded_files = []
        st.session_state.uploaded_names = []
        st.session_state.uploaded_types = []
        st.session_state.image_hash     = None

        # Force the uploader widget to reset by changing its key
        st.session_state.upload_key += 1

        # Run the caller's cleanup callback if one was provided
        if self.on_delete:
            self.on_delete()

        # Refresh the page immediately so the cleared state is reflected
        st.rerun()

    # -------------------------------------------------------------------------
    # HASHING
    # -------------------------------------------------------------------------

    def _compute_hash(self, images: list[bytes]) -> str | None:
        """Compute a single combined MD5 hash across all image byte strings.

        Concatenates all image bytes before hashing so that any change to
        any image — or a change in the set of images — produces a different
        hash value.

        Args:
            images: List of raw image byte strings.

        Returns:
            Hex MD5 hash string, or None if the list is empty.
        """
        if not images:
            return None

        # Concatenate all image bytes into a single stream before hashing
        combined = b"".join(images)
        return hashlib.md5(combined).hexdigest()

    # -------------------------------------------------------------------------
    # PUBLIC INTERFACE
    # -------------------------------------------------------------------------

    def get_images(self) -> list[bytes]:
        """Return all current image byte strings from session state.

        Returns:
            List of raw image bytes. Empty list if no images are uploaded.
        """
        return st.session_state.get("uploaded_files", [])

    def get_image(self) -> bytes | None:
        """Return the first image for backwards compatibility with single-image code.

        Returns:
            Raw bytes of the first image, or None if no images are uploaded.
        """
        images = self.get_images()
        return images[0] if images else None

    def get_hash(self) -> str | None:
        """Return the combined MD5 hash of all current images.

        Returns:
            Hex MD5 hash string, or None if no images are uploaded.
        """
        return st.session_state.get("image_hash")

    def is_new_image(self, last_hash: str | None) -> bool:
        """Check whether the current image set differs from the last processed set.

        Compares the combined hash of all current images against the hash
        from the last LLM call. Returns True if any image has changed,
        been added, or been removed.

        Args:
            last_hash: Hash from the last time the LLM was called.

        Returns:
            True if the image set is new or changed, False if unchanged.
        """
        current_hash = self.get_hash()

        # Only return True if there are images AND they have changed
        return current_hash is not None and current_hash != last_hash

    def get_types(self) -> list[str]:
        """Return all current image MIME types from session state."""
        return st.session_state.get("uploaded_types", [])

    # -------------------------------------------------------------------------
    # UI RENDERING
    # -------------------------------------------------------------------------

    def render(self):
        """Render the full upload UI.

        Displays:
        - Multi-file uploader widget
        - Validation feedback per file
        - Preview of all successfully uploaded images
        - Success message listing all filenames
        - Delete all button
        """
        # Multi-file uploader — dynamic key resets the widget when upload_key changes
        uploaded_files = st.file_uploader(
            "Upload image",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True,        # ← allow multiple images
            key=f"uploader_{st.session_state.upload_key}"
        )

        # If new files were uploaded, validate and save them to session state
        if uploaded_files:
            self._save_to_session(uploaded_files)

        # Display previews, filenames and delete button if images are in state
        if st.session_state.uploaded_files:
            
            # Show all images in a grid — 3 columns for compact display
            names  = st.session_state.uploaded_names
            images = st.session_state.uploaded_files
            count  = len(images)

            # Use up to 3 columns, fewer if less than 3 images
            n_cols = min(count, 3)
            cols   = st.columns(n_cols)

            for i, (img_bytes, name) in enumerate(zip(images, names)):
                with cols[i % n_cols]:
                    # Show image preview
                    st.image(img_bytes, use_container_width=True)
                    # Show filename below preview
                    st.caption(name)

            # Success message summarising all uploaded files
            names_str = ", ".join(names)
            st.success(f"✅ {count} image(s) uploaded: {names_str}")

            # Single delete button clears all images at once
            if st.button("🗑️ Delete image"):
                self._delete()

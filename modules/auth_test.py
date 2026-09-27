import streamlit as st

from modules.auth import (
    get_supabase_client,
    initialize_auth_state,
    is_authenticated,
)


st.set_page_config(page_title="Auth Test")

st.title("Authentication Test")

try:
    initialize_auth_state()

    client = get_supabase_client()

    st.success("Supabase client initialized successfully.")

    st.write("Authentication state initialized.")
    st.write("Currently authenticated:", is_authenticated())

    st.success("Step 6 test passed.")

except Exception as e:
    st.error("Authentication test failed.")
    st.exception(e)

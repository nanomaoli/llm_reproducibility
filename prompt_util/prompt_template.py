
def make_conversation_from_contents(
    contents,
    system_prompt=None,
    user_template=None,
    assistant_prefill=None,
):
    """Makes a conversation given a list of user/assistant message strings.

    If system_prompt is provided, it will be added as the first message.
    If user_template is provided, it will be used to format the user messages. This is useful for model-specific formatting.

    Args:
        content: A list of user/assistant message strings.
        system_prompt: An optional string for the system prompt.
        user_template: An optional string for the user template.

    Returns:
        A list of dictionaries representing the conversation.
    """

    conversation = []
    if system_prompt:
        conversation.append({"role": "system", "content": system_prompt})

    for i, content in enumerate(contents):
        if i % 2 == 0:
            content = user_template.format(content) if user_template else content
            conversation.append({"role": "user", "content": content})
        else:
            conversation.append({"role": "assistant", "content": content})

    if assistant_prefill and conversation[-1]["role"] == "user":
        conversation.append({"role": "assistant", "content": assistant_prefill})

    return conversation
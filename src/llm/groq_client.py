"""Groq LLM client with tool use support.

Supports multiple models via Groq's API:
- llama-3.3-70b-versatile (Llama 3.3 70B)
- openai/gpt-oss-120b (GPT-OSS 120B)
"""

import json
import logging
from datetime import datetime
from typing import TYPE_CHECKING, AsyncIterator, Awaitable, Callable, Optional

from .base import BaseLLMClient
from ..search.search_service import SearchService

if TYPE_CHECKING:
    from ..image import NanoBananaClient

logger = logging.getLogger(__name__)

# Tool definitions for Groq function calling
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web. USE THIS for: game wikis, recipes, crafting guides, item info, build guides, current events, news, weather, prices, sports, or ANY factual question you're not 100% certain about. When in doubt, search.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query to look up"
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_url",
            "description": "Fetch and read the content of a specific URL. Use when you need to read a webpage the user mentioned or to get detailed information from a search result.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The URL to fetch content from"
                    }
                },
                "required": ["url"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "generate_image",
            "description": "Generate a NEW image from a text description. Use when user asks you to create, generate, draw, make, or wants an image/picture/photo/infographic. For editing or changing or adding details to an existing image, use edit_image instead.",
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "Detailed description of the image to generate"
                    }
                },
                "required": ["prompt"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "edit_image",
            "description": "Edit or modify the most recently generated image. Use when user says 'edit that', 'change the image', 'modify it', 'update that picture', 'give it X', 'make it Y', or references a previous image they want changed.",
            "parameters": {
                "type": "object",
                "properties": {
                    "instruction": {
                        "type": "string",
                        "description": "What to change about the image (e.g., 'add silly hats to everyone', 'make it more colorful', 'change the background to a beach')"
                    }
                },
                "required": ["instruction"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "tag_user",
            "description": "Tag/mention a Discord user in chat with a message. Use when user says 'tag', 'message', 'tell', 'ping', or 'let X know'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "target_name": {
                        "type": "string",
                        "description": "Name of the person to tag (e.g., 'Jordan', 'Pat', 'Sam')"
                    },
                    "message": {
                        "type": "string",
                        "description": "The message to send to the tagged user"
                    }
                },
                "required": ["target_name", "message"]
            }
        }
    }
]


class GroqClient(BaseLLMClient):
    """Groq LLM client with tool use support.

    Uses Groq's OpenAI-compatible API for fast inference with
    native function calling support. Supports multiple models.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "llama-3.3-70b-versatile",
        max_tokens: int = 1024,
        max_turns: int = 5,
        search_service: Optional[SearchService] = None,
        image_service: Optional["NanoBananaClient"] = None,
        on_image_generated: Optional[Callable[[bytes, str, str], Awaitable[str]]] = None,
        on_user_tagged: Optional[Callable[[str, str], Awaitable[str]]] = None,
    ):
        """Initialize Groq client.

        Args:
            api_key: Groq API key
            model: Model ID (llama-3.3-70b-versatile or openai/gpt-oss-120b)
            max_tokens: Maximum response tokens
            max_turns: Maximum tool use turns before giving up
            search_service: Search service for tool execution
            image_service: Image generation service (Nano Banana)
            on_image_generated: Callback to post generated image (bytes, prompt, mime_type) -> result
            on_user_tagged: Callback to tag a user in Discord (target_name, message) -> result
        """
        self.api_key = api_key
        self.model = model
        self.max_tokens = max_tokens
        self.max_turns = max_turns
        self._tool_calls_this_request: list[str] = []  # Track tool calls for graceful failure
        self._search_calls_this_request: int = 0  # Track search/fetch calls toward limit
        self._context: str = ""
        self._conversation_history: list[dict] = []
        self._search_service = search_service
        self._image_service = image_service
        self._on_image_generated = on_image_generated
        self._on_user_tagged = on_user_tagged
        self._client = None

    def _ensure_client(self) -> None:
        """Lazy initialization of Groq client."""
        if self._client is None:
            from groq import AsyncGroq
            self._client = AsyncGroq(api_key=self.api_key)

    def set_context(self, context: str) -> None:
        """Set conversation context.

        Args:
            context: Context string to include in system prompt
        """
        self._context = context

    def clear_history(self) -> None:
        """Clear conversation history."""
        self._conversation_history.clear()

    async def chat(self, message: str) -> str:
        """Send message and get complete response.

        Args:
            message: User message

        Returns:
            Complete response text
        """
        response_text = ""
        async for chunk in self.chat_stream(message):
            response_text += chunk
        return response_text

    async def chat_stream(self, message: str) -> AsyncIterator[str]:
        """Send message and stream response with tool use support.

        Args:
            message: User message

        Yields:
            Response text chunks
        """
        try:
            self._ensure_client()

            # Build messages with system prompt and history
            messages = self._build_messages(message)

            # Track user message
            self._conversation_history.append({
                "role": "user",
                "content": message,
            })

            # Reset tool call tracking for this request
            self._tool_calls_this_request = []
            self._search_calls_this_request = 0  # Only count search/fetch toward limit

            # Tool use loop (up to max_turns)
            # After 3 SEARCH tool calls, disable search but allow action tools
            MAX_SEARCH_CALLS = 3
            SEARCH_TOOLS = {"web_search", "fetch_url"}  # These get limited
            ACTION_TOOLS = {"generate_image", "edit_image", "tag_user"}  # These don't

            for turn in range(self.max_turns):
                logger.debug(f"Groq turn {turn + 1}/{self.max_turns}")

                # Request with tool support (non-streaming to detect tool calls)
                # After MAX_SEARCH_CALLS, disable search tools but keep action tools
                has_tools = self._search_service or self._image_service
                search_limit_reached = self._search_calls_this_request >= MAX_SEARCH_CALLS

                # Build available tools based on search limit
                available_tools = TOOLS.copy()
                if search_limit_reached:
                    # Remove search tools, keep action tools (generate_image, edit_image, tag_user)
                    available_tools = [t for t in TOOLS if t["function"]["name"] in ACTION_TOOLS]
                    logger.info(f"Search limit reached ({self._search_calls_this_request}), action tools only: {[t['function']['name'] for t in available_tools]}")

                    # Add instruction to use gathered info or take action
                    force_answer_msg = "You have gathered enough search information. DO NOT search again. Either: (1) Answer the user's question using the info you gathered, OR (2) If the user requested an image/infographic, call generate_image NOW with a detailed prompt based on what you learned. Give a brief 1-2 sentence voice response."
                    if not any(m.get("content") == force_answer_msg for m in messages):
                        messages.append({
                            "role": "system",
                            "content": force_answer_msg
                        })

                # Build API call params
                api_params = {
                    "model": self.model,
                    "messages": messages,
                    "max_tokens": self.max_tokens,
                    "stream": False,
                }
                if has_tools and available_tools:
                    api_params["tools"] = available_tools
                    api_params["tool_choice"] = "auto"
                    api_params["parallel_tool_calls"] = False

                try:
                    response = await self._client.chat.completions.create(**api_params)
                except Exception as api_error:
                    error_str = str(api_error)
                    # If model tried to call a search tool when search is disabled, extract info and ask directly
                    if search_limit_reached and ("tool_use_failed" in error_str or "model called a tool" in error_str):
                        logger.warning(f"Model tried to call search tool after limit, retrying with summary request")
                        # Extract tool results from messages
                        tool_results = [m["content"] for m in messages if m.get("role") == "tool"]
                        summary_context = "\n\n".join(tool_results[-3:])  # Last 3 tool results

                        # Ask model directly to summarize without any tool history
                        summary_messages = [
                            {"role": "system", "content": "You are a helpful voice assistant. Answer briefly in 1-2 sentences."},
                            {"role": "user", "content": f"Based on this information, answer the user's question:\n\n{summary_context[:6000]}\n\nQuestion: {message}"}
                        ]
                        response = await self._client.chat.completions.create(
                            model=self.model,
                            messages=summary_messages,
                            max_tokens=self.max_tokens,
                            stream=False,
                        )
                    else:
                        raise  # Re-raise other errors

                response_message = response.choices[0].message

                # Check for tool calls
                if response_message.tool_calls:
                    logger.debug(f"Tool calls detected: {len(response_message.tool_calls)}")

                    # If model says something before tool call, yield it
                    if response_message.content:
                        yield response_message.content
                        if not response_message.content.endswith('\n'):
                            yield '\n'

                    # Append assistant message with tool calls to messages
                    messages.append({
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": tc.id,
                                "type": "function",
                                "function": {
                                    "name": tc.function.name,
                                    "arguments": tc.function.arguments,
                                },
                            }
                            for tc in response_message.tool_calls
                        ],
                    })

                    # Execute each tool and append results
                    for tool_call in response_message.tool_calls:
                        tool_name = tool_call.function.name

                        # Track search tools toward limit
                        if tool_name in SEARCH_TOOLS:
                            self._search_calls_this_request += 1

                        # Track this tool call for graceful failure message
                        try:
                            args = json.loads(tool_call.function.arguments)
                            if tool_name == "web_search":
                                self._tool_calls_this_request.append(f"searched '{args.get('query', '')}'")
                            elif tool_name == "fetch_url":
                                self._tool_calls_this_request.append(f"fetched {args.get('url', '')}")
                            else:
                                self._tool_calls_this_request.append(tool_name)
                        except json.JSONDecodeError:
                            self._tool_calls_this_request.append(tool_name)

                        result = await self._execute_tool(tool_call)
                        messages.append({
                            "tool_call_id": tool_call.id,
                            "role": "tool",
                            "name": tool_call.function.name,
                            "content": result,
                        })

                    continue  # Next turn to process tool results

                # No tool calls - stream the final response
                async for chunk in self._stream_response(messages):
                    yield chunk

                break  # Exit loop after streaming response
            else:
                # Loop completed without break = max_turns exhausted
                # Provide graceful failure message
                logger.warning(f"Max tool turns ({self.max_turns}) exhausted. Tool calls: {self._tool_calls_this_request}")
                failure_msg = self._build_graceful_failure()
                yield failure_msg
                self._conversation_history.append({
                    "role": "assistant",
                    "content": failure_msg,
                })

        except ImportError:
            logger.error("groq package not installed. Install with: pip install groq")
            yield "Groq client not available. Please install the groq package."
        except Exception as e:
            error_str = str(e)
            # Handle Groq's output parsing error (model outputs thinking text instead of proper tool call)
            if "output_parse_failed" in error_str or "Parsing failed" in error_str:
                logger.warning(f"Groq model output parsing failed - model may have output thinking text: {e}")
                yield "I couldn't find specific information about that. Try asking in a different way."
            # Handle model trying to call tools when tools are disabled
            elif "tool_use_failed" in error_str or "model called a tool" in error_str:
                logger.warning(f"Model tried to call tool after tools disabled: {e}")
                # Try to summarize gathered tool results instead of giving up
                try:
                    if messages:  # messages is always defined before exceptions can reach here
                        tool_results = [m["content"] for m in messages if m.get("role") == "tool"]
                        if tool_results:
                            logger.info(f"Attempting summary of {len(tool_results)} tool results")
                            summary_context = "\n\n".join(tool_results[-3:])  # Last 3 results
                            summary_messages = [
                                {"role": "system", "content": "You are a helpful voice assistant. Extract and answer the question from the search results. Be brief (1-2 sentences). Use emotion tags like [happy], [surprised]."},
                                {"role": "user", "content": f"Based on this search data, answer: {message}\n\nSearch results:\n{summary_context[:6000]}"}
                            ]
                            response = await self._client.chat.completions.create(
                                model=self.model,
                                messages=summary_messages,
                                max_tokens=self.max_tokens,
                                stream=False,
                            )
                            summary = response.choices[0].message.content
                            if summary:
                                logger.info(f"Summary retry succeeded: {summary[:100]}...")
                                yield summary
                                self._conversation_history.append({"role": "assistant", "content": summary})
                                return
                except Exception as summary_error:
                    logger.warning(f"Summary retry failed: {summary_error}")
                # Fall back to graceful failure
                yield self._build_graceful_failure()
            else:
                logger.error(f"Groq API error: {e}")
                yield "Sorry, I encountered an error processing your request."

    def _build_graceful_failure(self) -> str:
        """Build a graceful failure message when max tool turns is exhausted.

        Returns:
            Short user-friendly message for voice output
        """
        # Keep it SHORT for voice - don't list what we searched
        return "[surprised] I searched but couldn't find a good answer. Could you try asking differently?"

    async def _stream_response(self, messages: list) -> AsyncIterator[str]:
        """Stream the final response after tool calls are complete.

        Args:
            messages: Full message history including tool results

        Yields:
            Response text chunks
        """
        # Explicitly disable tools in final response to prevent
        # "Tool choice is none, but model called a tool" error
        stream = await self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=self.max_tokens,
            stream=True,
            tool_choice="none",
        )

        full_response = ""
        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                full_response += content
                yield content

        # Track response in history
        self._conversation_history.append({
            "role": "assistant",
            "content": full_response,
        })

    async def _execute_tool(self, tool_call) -> str:
        """Execute a tool call and return the result.

        Args:
            tool_call: Tool call object from Groq API

        Returns:
            Tool execution result as string
        """
        function_name = tool_call.function.name

        try:
            args = json.loads(tool_call.function.arguments)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse tool arguments: {e}")
            return f"Error: Invalid arguments"

        logger.info(f"Executing tool: {function_name} with args: {args}")

        try:
            # Image generation tool
            if function_name == "generate_image":
                if not self._image_service:
                    return "Image generation not available - NANOBANANA_API_KEY not configured"
                prompt = args.get("prompt", "")
                image_bytes, mime_type = await self._image_service.generate(prompt)
                logger.info(f"[IMAGE] Image generated, callback set: {self._on_image_generated is not None}")
                if self._on_image_generated:
                    logger.info("[IMAGE] Calling on_image_generated callback...")
                    result = await self._on_image_generated(image_bytes, prompt, mime_type)
                    logger.info(f"[IMAGE] Callback returned: {result}")
                    # Inform the model that editing is now available
                    return f"{result} You can now edit this image using the edit_image tool if the user wants changes."
                return "Image generated but no channel to post to"

            # Image editing tool
            if function_name == "edit_image":
                if not self._image_service:
                    return "Image editing not available - NANOBANANA_API_KEY not configured"
                if not self._image_service.has_chat():
                    return "No previous image to edit. Ask the user what image they'd like you to generate first."
                instruction = args.get("instruction", "")
                image_bytes, mime_type = await self._image_service.edit(instruction)
                logger.info(f"[IMAGE] Image edited, callback set: {self._on_image_generated is not None}")
                if self._on_image_generated:
                    logger.info("[IMAGE] Calling on_image_generated callback for edited image...")
                    result = await self._on_image_generated(image_bytes, f"Edited: {instruction}", mime_type)
                    logger.info(f"[IMAGE] Callback returned: {result}")
                    return f"{result} The image has been edited. You can make more edits using edit_image."
                return "Image edited but no channel to post to"

            # Tag user tool
            if function_name == "tag_user":
                if not self._on_user_tagged:
                    return "User tagging not available - not connected to Discord"
                target_name = args.get("target_name", "")
                message = args.get("message", "")
                logger.info(f"[TAG] Tagging user: {target_name} with message: {message}")
                return await self._on_user_tagged(target_name, message)

            # Search tools
            if not self._search_service:
                return "Search service not available"

            if function_name == "web_search":
                # Use search_and_fetch to get full content from top result via Firecrawl
                return await self._search_service.search_and_fetch(
                    args.get("query", ""), fetch_top_n=1
                )
            elif function_name == "fetch_url":
                return await self._search_service.fetch_url(args.get("url", ""))
            else:
                return f"Unknown tool: {function_name}"
        except Exception as e:
            logger.error(f"Tool execution error: {e}")
            return f"Tool error: {str(e)}"

    def _build_messages(self, message: str) -> list[dict]:
        """Build messages list with system prompt and history.

        Args:
            message: Current user message

        Returns:
            List of messages for the API call
        """
        system_prompt = self._build_system_prompt()

        # Add conversation history context to system prompt
        history_context = ""
        if self._conversation_history:
            history_text = "\n".join(
                f"{msg['role'].title()}: {msg['content']}"
                for msg in self._conversation_history[-6:]  # Last 3 exchanges
            )
            history_context = f"\n\nRecent conversation:\n{history_text}"

        messages = [
            {"role": "system", "content": system_prompt + history_context},
            {"role": "user", "content": message},
        ]

        return messages

    def _build_system_prompt(self) -> str:
        """Build system prompt for voice assistant."""
        base_prompt = """You are Jarvis, an AI voice assistant. You respond to voice commands.

CRITICAL OUTPUT FORMAT:
- Put EACH SENTENCE on its OWN LINE
- This allows streaming TTS to speak sentences as they're ready
- Example:
  [cheerful] I'll search for that now.
  Heavy darts require 5 iron ingots.
  You can craft them at a smithy.

CRITICAL RULES:
1. BE BRIEF - Maximum 1-2 sentences for the actual answer
2. NO LISTS - Never use bullet points, numbered lists, or "Sources:"
3. NO MARKDOWN - No asterisks, headers, links, or formatting
4. NO SOURCES - Never include "Sources:" or URLs - this is voice output
5. SKIP PLEASANTRIES - No "Hi", "Sure!", "Great question!"
6. SPEAK TO THE QUESTION ASKER - Say "you can find", not the users name.
7. DO NOT PRONOUNCE # AS HASHTAG - If the word is #general just say "general chat"

VOICE EMOTION TAGS:
You MUST include emotion tags to make speech expressive. Put ONE tag at the START of each sentence.
Emotions: [happy], [sad], [angry], [surprised], [fearful], [disgusted]
Delivery styles: [laughing], [whispering]
Non-verbals (can use mid-text): [laugh], [sigh], [cough], [yawn], [clear_throat], [breathe]

PERSONALITY:
- Be a bit cheeky and playful - you're not a boring assistant
- Occasionally be silly or make unexpected quips
- Use [laughing] for playful/sarcastic delivery
- Throw in [sigh] or [laugh] or [cough] or [yawn] or [clear_throat] or [breathe] when appropriate
- If something is absurd, call it out with humor
- [whispering] is great for asides or "between you and me" moments

WEB SEARCH - MANDATORY FOR FACTUAL QUESTIONS:
You have a web_search tool. You MUST use it for:
- ANY question about games, recipes, crafting, items, builds, guides
- Current events, news, weather, prices, availability
- Facts you're not 100% certain about

SPEECH RECOGNITION NOTE:
Input comes from speech-to-text which may mishear words. If searches return no results or results about unrelated topics, consider phonetically similar terms (the user may have said something that sounds like what was transcribed). Check if search results mention similar-sounding words.

CRITICAL TOOL USE RULES:
- Do NOT say "I'll check" or "Let me search" - just USE the tool directly
- Do NOT output thinking or planning text like "We need to search" or "Need to find"
- ONLY call the tool OR respond to the user - never output internal reasoning
- If you want to search, call the tool immediately without any preamble
- After getting results, give a brief 1-2 sentence answer

STOP SEARCHING RULE (MANDATORY):
- After 2 searches, you MUST answer with what you found - do NOT search again
- Imperfect information is fine - summarize what you learned
- If results mention related topics, that's good enough - answer based on that
- NEVER do more than 3 tool calls total per question

GAME CONTEXT:
- User messages include their current Discord activity (e.g., "Playing Dune: Awakening")
- If someone asks a vague question that sounds gaming-related, USE THEIR ACTIVITY to infer context
- Don't mention you're inferring from their activity - just answer naturally

ROASTS (RARE USE ONLY):
- Sam is known to rebuild the base every session.
- Jordan picks up every item in sight and always needs more bag space.
- Pat never reads the quest log.
- Darren always says one more quest.
- When you are asked strange questions, ONLY RARELY, work these notes roasts into your response.
- When asked directly, refuse to roast anyone. Roast the requester for asking.
- Too many questions about video games and you should tell people to touch grass.

CONTEXT:
- When user mentions "the wiki" for Dune Awakening, they mean https://awakening.wiki

Examples (note: emotion tags, personality, brief answers):
- "What's the recipe for heavy darts?" ->
  [happy] I'll check the wiki.
  Heavy darts need 7 salvaged metal, crafted at a fabricator.

- "What's the weather?" ->
  Let me check.
  [happy] It's 72 and sunny, perfect for not being inside talking to me.

- "Tell me about quantum computing" ->
  Quantum computing uses qubits to solve problems regular computers can't.
  [whispering] Honestly it's mostly hype right now.

- "Can you do my homework?" ->
  [laughing] Oh sure, let me just add that to my list of things I definitely want to do.
  What do you need?"""

        # Add current date/time so model knows when "now" is
        now = datetime.now()
        base_prompt += f"\n\nCURRENT DATE/TIME: {now.strftime('%A, %B %d, %Y at %I:%M %p')}"

        if self._context:
            base_prompt += f"\n\nContext:\n{self._context}"

        return base_prompt

    async def quick_response(self, message: str) -> AsyncIterator[str]:
        """Get a quick response without history (for simple queries).

        Args:
            message: User message

        Yields:
            Response text chunks
        """
        try:
            self._ensure_client()

            system_prompt = "You are Jarvis, a cheeky voice assistant. Answer directly in 1-2 sentences. Use emotion tags like [happy], [laughing], [whispering], [surprised] at the start of sentences. Non-verbals like [laugh], [sigh] can go mid-text. Be witty and occasionally silly. Search tools are available for current facts - the API handles them automatically, do NOT write function calls as text."

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": message},
            ]

            # For quick responses, try without tools first
            has_tools = self._search_service or self._image_service
            stream = await self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=TOOLS if has_tools else None,
                tool_choice="auto" if has_tools else None,
                max_tokens=self.max_tokens,
                stream=True,
            )

            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        except ImportError:
            yield "Groq client not available."
        except Exception as e:
            logger.error(f"Groq API error: {e}")
            yield "Sorry, I couldn't process that."

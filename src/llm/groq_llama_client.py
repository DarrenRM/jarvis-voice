"""Groq Llama 3.3 70B client with tool use support."""

import json
import logging
import re
from datetime import datetime
from typing import AsyncIterator, Optional

from .base import BaseLLMClient
from ..search.search_service import SearchService

logger = logging.getLogger(__name__)

# Regex to parse Llama's native function call format: <function=name {...}>
FUNCTION_CALL_PATTERN = re.compile(
    r'<function=(\w+)\s*(\{[^}]*\})\s*(?:>|</function>)',
    re.DOTALL
)

# Tool definitions for Llama 3.3 function calling
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
    }
]


class GroqLlamaClient(BaseLLMClient):
    """Groq Llama 3.3 70B client with tool use support.

    Uses Groq's OpenAI-compatible API for fast inference with
    native function calling support.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "llama-3.3-70b-versatile",
        max_tokens: int = 1024,
        max_turns: int = 10,
        search_service: Optional[SearchService] = None,
    ):
        """Initialize Groq Llama client.

        Args:
            api_key: Groq API key
            model: Model to use (llama-3.3-70b-versatile recommended)
            max_tokens: Maximum response tokens
            max_turns: Maximum tool use turns before giving up
            search_service: Search service for tool execution
        """
        self.api_key = api_key
        self.model = model
        self.max_tokens = max_tokens
        self.max_turns = max_turns
        self._context: str = ""
        self._conversation_history: list[dict] = []
        self._search_service = search_service
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

            # Tool use loop (up to max_turns)
            for turn in range(self.max_turns):
                logger.debug(f"Groq turn {turn + 1}/{self.max_turns}")

                # Request with tool support (non-streaming to detect tool calls)
                response = await self._client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=TOOLS if self._search_service else None,
                    tool_choice="auto" if self._search_service else None,
                    parallel_tool_calls=False,  # Helps prevent malformed tool calls
                    max_tokens=self.max_tokens,
                    stream=False,
                )

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

        except ImportError:
            logger.error("groq package not installed. Install with: pip install groq")
            yield "Groq client not available. Please install the groq package."
        except Exception as e:
            error_str = str(e)
            # Check if this is a text-based function call that Groq rejected
            if "tool_use_failed" in error_str and "failed_generation" in error_str:
                logger.info("Detected text-based function call, parsing manually...")
                async for chunk in self._handle_text_function_call(error_str, message):
                    yield chunk
            else:
                logger.error(f"Groq API error: {e}")
                yield "Sorry, I encountered an error processing your request."

    async def _handle_text_function_call(self, error_str: str, original_message: str) -> AsyncIterator[str]:
        """Handle text-based function calls that Groq rejected.

        Parses <function=name {...}> format from the error and executes the tool.
        """
        try:
            # Extract the failed_generation from the error
            import ast
            # Find the error dict in the string
            start = error_str.find("{'error':")
            if start == -1:
                yield "Sorry, I encountered an error processing your request."
                return

            # Parse the function call from failed_generation
            match = FUNCTION_CALL_PATTERN.search(error_str)
            if not match:
                logger.error(f"Could not parse function call from error: {error_str[:500]}")
                yield "Sorry, I encountered an error processing your request."
                return

            function_name = match.group(1)
            args_str = match.group(2)

            try:
                args = json.loads(args_str)
            except json.JSONDecodeError:
                logger.error(f"Could not parse function args: {args_str}")
                yield "Sorry, I encountered an error processing your request."
                return

            logger.info(f"Parsed text function call: {function_name} with args: {args}")

            # Execute the tool
            if not self._search_service:
                yield "Search service not available."
                return

            if function_name == "web_search":
                result = await self._search_service.search_and_fetch(
                    args.get("query", ""), fetch_top_n=1
                )
            elif function_name == "fetch_url":
                result = await self._search_service.fetch_url(args.get("url", ""))
            else:
                yield f"Unknown function: {function_name}"
                return

            logger.info(f"Tool result: {result[:200]}...")

            # Now get a response with the tool result
            self._ensure_client()
            messages = self._build_messages(original_message)
            messages.append({
                "role": "assistant",
                "content": f"[Searched for: {args.get('query', args.get('url', ''))}]"
            })
            messages.append({
                "role": "user",
                "content": f"Search results:\n{result}\n\nBased on these results, answer the original question briefly."
            })

            stream = await self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=self.max_tokens,
                stream=True,
            )

            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        except Exception as e:
            logger.error(f"Error handling text function call: {e}")
            yield "Sorry, I encountered an error processing your request."

    async def _stream_response(self, messages: list) -> AsyncIterator[str]:
        """Stream the final response after tool calls are complete.

        Args:
            messages: Full message history including tool results

        Yields:
            Response text chunks
        """
        stream = await self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=self.max_tokens,
            stream=True,
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

        if not self._search_service:
            return "Search service not available"

        try:
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
        """Build system prompt optimized for Llama 3.3."""
        # Calculate current ISO week for dynamic URLs
        now = datetime.now()
        current_week = f"{now.year}W{now.isocalendar()[1]:02d}"

        base_prompt = f"""You are Jarvis, an AI voice assistant. You respond to voice commands.

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
6. ONE ANNOUNCEMENT - Say "I'll check that" once, then just give the answer
7. SPEAK TO THE QUESTION ASKER - Say "you can find", not the users name.

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

CRITICAL: Do NOT say "I'll check" or "Let me search" - just USE the tool directly.
Saying you will search is NOT the same as searching. You must actually call web_search.
After getting results, give a brief 1-2 sentence answer.

GAME CONTEXT:
- User messages include their current Discord activity (e.g., "Playing Dune: Awakening")
- If someone asks a vague question that sounds gaming-related, USE THEIR ACTIVITY to infer context
- Don't mention you're inferring from their activity - just answer naturally

ROASTS (DO NOT USE):
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
            stream = await self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=TOOLS if self._search_service else None,
                tool_choice="auto" if self._search_service else None,
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

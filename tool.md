Writing A Custom Toolkit
Toolkits are defined in a single Python file, with a top level docstring with metadata and a Tools class.

Example Top-Level Docstring
"""
title: String Inverse
author: Your Name
author_url: https://website.com
git_url: https://github.com/username/string-reverse.git
description: This tool calculates the inverse of a string
required_open_webui_version: 0.4.0
requirements: langchain-openai, langgraph, ollama, langchain_ollama
version: 0.4.0
licence: MIT
"""

Tools Class
Tools have to be defined as methods within a class called Tools, with optional subclasses called Valves and UserValves, for example:

class Tools:
    def __init__(self):
        """Initialize the Tool."""
        self.valves = self.Valves()

    class Valves(BaseModel):
        api_key: str = Field("", description="Your API key here")

    def reverse_string(self, string: str) -> str:
        """
        Reverses the input string.
        :param string: The string to reverse
        """
        # example usage of valves
        if self.valves.api_key != "42":
            return "Wrong API key"
        return string[::-1] 

Type Hints
Each tool must have type hints for arguments. The types may also be nested, such as queries_and_docs: list[tuple[str, int]]. Those type hints are used to generate the JSON schema that is sent to the model. Tools without type hints will work with a lot less consistency.

Valves and UserValves - (optional, but HIGHLY encouraged)
Valves and UserValves are used for specifying customizable settings of the Tool, you can read more on the dedicated Valves & UserValves page.

Optional Arguments
Below is a list of optional arguments your tools can depend on:

__event_emitter__: Emit events (see following section)
__event_call__: Same as event emitter but can be used for user interactions
__user__: A dictionary with user information. It also contains the UserValves object in __user__["valves"].
__metadata__: Dictionary with chat metadata
__messages__: List of previous messages
__files__: Attached files
__model__: A dictionary with model information
__oauth_token__: A dictionary containing the user's valid, automatically refreshed OAuth token payload. This is the new, recommended, and secure way to access user tokens for making authenticated API calls. The dictionary typically contains access_token, id_token, and other provider-specific data.
For more information about __oauth_token__ and how to configure this token to be sent to tools, check out the OAuth section in the environment variable docs page and the SSO documentation.

Just add them as argument to any method of your Tool class just like __user__ in the example above.

Using the OAuth Token in a Tool
When building tools that need to interact with external APIs on the user's behalf, you can now directly access their OAuth token. This removes the need for fragile cookie scraping and ensures the token is always valid.

Example: A tool that calls an external API using the user's access token.

import httpx
from typing import Optional

class Tools:
    # ... other class setup ...

    async def get_user_profile_from_external_api(self, __oauth_token__: Optional[dict] = None) -> str:
        """
        Fetches user profile data from a secure external API using their OAuth access token.

        :param __oauth_token__: Injected by Open WebUI, contains the user's token data.
        """
        if not __oauth_token__ or "access_token" not in __oauth_token__:
            return "Error: User is not authenticated via OAuth or token is unavailable."
            
        access_token = __oauth_token__["access_token"]
        
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json"
        }
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get("https://api.my-service.com/v1/profile", headers=headers)
                response.raise_for_status() # Raise an exception for bad status codes
                return f"API Response: {response.json()}"
        except httpx.HTTPStatusError as e:
            return f"Error: Failed to fetch data from API. Status: {e.response.status_code}"
        except Exception as e:
            return f"An unexpected error occurred: {e}"


Event Emitters
Event Emitters are used to add additional information to the chat interface. Similarly to Filter Outlets, Event Emitters are capable of appending content to the chat. Unlike Filter Outlets, they are not capable of stripping information. Additionally, emitters can be activated at any stage during the Tool.

There are two different types of Event Emitters:

If the model seems to be unable to call the tool, make sure it is enabled (either via the Model page or via the + sign next to the chat input field). You can also turn the Function Calling argument of the Advanced Params section of the Model page from Default to Native.

Status
This is used to add statuses to a message while it is performing steps. These can be done at any stage during the Tool. These statuses appear right above the message content. These are very useful for Tools that delay the LLM response or process large amounts of information. This allows you to inform users what is being processed in real-time.

await __event_emitter__(
            {
                "type": "status", # We set the type here
                "data": {"description": "Message that shows up in the chat", "done": False, "hidden": False}, 
                # Note done is False here indicating we are still emitting statuses
            }
        )


Example
async def test_function(
        self, prompt: str, __user__: dict, __event_emitter__=None
    ) -> str:
        """
        This is a demo

        :param test: this is a test parameter
        """

        await __event_emitter__(
            {
                "type": "status", # We set the type here
                "data": {"description": "Message that shows up in the chat", "done": False}, 
                # Note done is False here indicating we are still emitting statuses
            }
        )

        # Do some other logic here
        await __event_emitter__(
            {
                "type": "status",
                "data": {"description": "Completed a task message", "done": True, "hidden": False},
                # Note done is True here indicating we are done emitting statuses
                # You can also set "hidden": True if you want to remove the status once the message is returned
            }
        )

        except Exception as e:
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {"description": f"An error occured: {e}", "done": True},
                }
            )

            return f"Tell the user: {e}"

Message
This type is used to append a message to the LLM at any stage in the Tool. This means that you can append messages, embed images, and even render web pages before, or after, or during the LLM response.

await __event_emitter__(
                    {
                        "type": "message", # We set the type here
                        "data": {"content": "This message will be appended to the chat."},
                        # Note that with message types we do NOT have to set a done condition
                    }
                )


Example
async def test_function(
        self, prompt: str, __user__: dict, __event_emitter__=None
    ) -> str:
        """
        This is a demo

        :param test: this is a test parameter
        """

        await __event_emitter__(
                    {
                        "type": "message", # We set the type here
                        "data": {"content": "This message will be appended to the chat."},
                        # Note that with message types we do NOT have to set a done condition
                    }
                )

        except Exception as e:
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {"description": f"An error occured: {e}", "done": True},
                }
            )

            return f"Tell the user: {e}"

Citations
This type is used to provide citations or references in the chat. You can utilize it to specify the content, the source, and any relevant metadata. Below is an example of how to emit a citation event:

await __event_emitter__(
    {
        "type": "citation",
        "data": {
            "document": [content],
            "metadata": [
                {
                    "date_accessed": datetime.now().isoformat(),
                    "source": title,
                }
            ],
            "source": {"name": title, "url": url},
        },
    }
)

If you are sending multiple citations, you can iterate over citations and call the emitter multiple times. When implementing custom citations, ensure that you set self.citation = False in your Tools class __init__ method. Otherwise, the built-in citations will override the ones you have pushed in. For example:

def __init__(self):
    self.citation = False

Warning: if you set self.citation = True, this will replace any custom citations you send with the automatically generated return citation. By disabling it, you can fully manage your own citation references.

Example
class Tools:
    class UserValves(BaseModel):
        test: bool = Field(
            default=True, description="test"
        )

    def __init__(self):
        self.citation = False

async def test_function(
        self, prompt: str, __user__: dict, __event_emitter__=None
    ) -> str:
        """
        This is a demo that just creates a citation

        :param test: this is a test parameter
        """

        await __event_emitter__(
            {
                "type": "citation",
                "data": {
                    "document": ["This message will be appended to the chat as a citation when clicked into"],
                    "metadata": [
                        {
                            "date_accessed": datetime.now().isoformat(),
                            "source": title,
                        }
                    ],
                    "source": {"name": "Title of the content", "url": "http://link-to-citation"},
                },
            }
        )

External packages
In the Tools definition metadata you can specify custom packages. When you click Save the line will be parsed and pip install will be run on all requirements at once.

Keep in mind that as pip is used in the same process as Open WebUI, the UI will be completely unresponsive during the installation.

No measures are taken to handle package conflicts with Open WebUI's requirements. That means that specifying requirements can break Open WebUI if you're not careful. You might be able to work around this by specifying open-webui itself as a requirement.

Example
"""
title: myToolName
author: myName
funding_url: [any link here will be shown behind a `Heart` button for users to show their support to you]
version: 1.0.0
# the version is displayed in the UI to help users keep track of updates.
license: GPLv3
description: [recommended]
requirements: package1>=2.7.0,package2,package3
"""
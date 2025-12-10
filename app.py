import re
from pytube import YouTube
from langchain_core.tools import tool
from IPython.display import display, JSON
import yt_dlp
from typing import List, Dict
from langchain_core.messages import HumanMessage
from langchain_core.messages import ToolMessage
import json
import config

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

# Suppress pytube errors
import logging
pytube_logger = logging.getLogger('pytube')
pytube_logger.setLevel(logging.ERROR)

# Suppress yt-dlp warnings
yt_dpl_logger = logging.getLogger('yt_dlp')
yt_dpl_logger.setLevel(logging.ERROR)

from langchain.chat_models import init_chat_model

llm = init_chat_model("gpt-4o-mini", model_provider="openai")

@tool
def extract_video_id(url: str) -> str:
    """
    Extracts the 11-character YouTube video ID from a URL.
    
    Args:
        url (str): A YouTube URL containing a video ID.

    Returns:
        str: Extracted video ID or error message if parsing fails.
    """
    
    # Regex pattern to match video IDs
    pattern = r'(?:v=|be/|embed/)([a-zA-Z0-9_-]{11})'
    match = re.search(pattern, url)
    return match.group(1) if match else "Error: Invalid YouTube URL"

print(extract_video_id.name)
print("----------------------------")
print(extract_video_id.description)
print("----------------------------")
print(extract_video_id.func)

print(extract_video_id.run("https://www.youtube.com/watch?v=hfIUstzHs9A"))

tools = []
tools.append(extract_video_id)

from youtube_transcript_api import YouTubeTranscriptApi

@tool
def fetch_transcript(video_id: str, language: str = "en") -> str:
    """
    Fetches the transcript of a YouTube video.
    
    Args:
        video_id (str): The YouTube video ID (e.g., "dQw4w9WgXcQ").
        language (str): Language code for the transcript (e.g., "en", "es").
    
    Returns:
        str: The transcript text or an error message.
    """
    
    try:
        ytt_api = YouTubeTranscriptApi()
        transcript = ytt_api.fetch(video_id, languages=[language])
        return " ".join([snippet.text for snippet in transcript.snippets])
    except Exception as e:
        return f"Error: {str(e)}"
    

print(fetch_transcript.run("hfIUstzHs9A"))

tools.append(fetch_transcript)

from pytube import Search
from langchain.tools import tool
from typing import List, Dict

@tool
def search_youtube(query: str) -> List[Dict[str, str]]:
    """
    Search YouTube for videos matching the query.
    
    Args:
        query (str): The search term to look for on YouTube
        
    Returns:
        List of dictionaries containing video titles and IDs in format:
        [{'title': 'Video Title', 'video_id': 'abc123'}, ...]
        Returns error message if search fails
    """
    try:
        s = Search(query)
        return [
            {
                "title": yt.title,
                "video_id": yt.video_id,
                "url": f"https://youtu.be/{yt.video_id}"
            }
            for yt in s.results
        ]
    except Exception as e:
        return f"Error: {str(e)}"
    
search_out=search_youtube.run("Generative AI")

display(search_out)

tools.append(search_youtube)

@tool
def get_full_metadata(url: str) -> dict:
    """Extract metadata given a YouTube URL, including title, views, duration, channel, likes, comments, and chapters."""
    with yt_dlp.YoutubeDL({'quiet': True, 'logger': yt_dpl_logger}) as ydl:
        info = ydl.extract_info(url, download=False)
        return {
            'title': info.get('title'),
            'views': info.get('view_count'),
            'duration': info.get('duration'),
            'channel': info.get('uploader'),
            'likes': info.get('like_count'),
            'comments': info.get('comment_count'),
            'chapters': info.get('chapters', [])
        }
        
meta_data=get_full_metadata.run("https://www.youtube.com/watch?v=T-D1OfcDW1M")

display(meta_data)

tools.append(get_full_metadata)

@tool
def get_thumbnails(url: str) -> List[Dict]:
    """
    Get available thumbnails for a YouTube video using its URL.
    
    Args:
        url (str): YouTube video URL (any format)
        
    Returns:
        List of dictionaries with thumbnail URLs and resolutions in YouTube's native order
    """
    
    try:
        with yt_dlp.YoutubeDL({'quiet': True, 'logger': yt_dpl_logger}) as ydl:
            info = ydl.extract_info(url, download=False)
            
            thumbnails = []
            for t in info.get('thumbnails', []):
                if 'url' in t:
                    thumbnails.append({
                        "url": t['url'],
                        "width": t.get('width'),
                        "height": t.get('height'),
                        "resolution": f"{t.get('width', '')}x{t.get('height', '')}".strip('x')
                    })
            
            return thumbnails

    except Exception as e:
        return [{"error": f"Failed to get thumbnails: {str(e)}"}]
    

thumbnails=get_thumbnails.run("https://www.youtube.com/watch?v=qWHaMrR5WHQ")

display(thumbnails)

tools.append(get_thumbnails)

llm_with_tools = llm.bind_tools(tools)

for tool in tools:
    schema = {
   "name": tool.name,
   "description": tool.description,
   "parameters": tool.args_schema.schema() if tool.args_schema else {},
   "return": tool.return_type if hasattr(tool, "return_type") else None}
    display(schema)

query = "I want to summarize youtube video: https://www.youtube.com/watch?v=T-D1OfcDW1M in english"
print(query)

messages = [HumanMessage(content = query)]
print(messages)

response_1 = llm_with_tools.invoke(messages)
response_1

messages.append(response_1)

tool_mapping = {
    "get_thumbnails" : get_thumbnails,
    "extract_video_id": extract_video_id,
    "fetch_transcript": fetch_transcript,
    "search_youtube": search_youtube,
    "get_full_metadata": get_full_metadata
}

tool_calls_1 = response_1.tool_calls
display(tool_calls_1)

tool_name=tool_calls_1[0]['name']
print(tool_name)

tool_call_id =tool_calls_1[0]['id']
print(tool_call_id)

args=tool_calls_1[0]['args']
print(args)

my_tool=tool_mapping[tool_calls_1[0]['name']]

video_id =my_tool.invoke(tool_calls_1[0]['args'])
video_id

messages.append(ToolMessage(content = video_id, tool_call_id = tool_calls_1[0]['id']))

response_2 = llm_with_tools.invoke(messages)
response_2

messages.append(response_2)

tool_calls_2 = response_2.tool_calls
tool_calls_2

fetch_transcript_tool_output = tool_mapping[tool_calls_2[0]['name']].invoke(tool_calls_2[0]['args'])
fetch_transcript_tool_output

messages.append(ToolMessage(content = fetch_transcript_tool_output, tool_call_id = tool_calls_2[0]['id']))

summary = llm_with_tools.invoke(messages)
print(summary)
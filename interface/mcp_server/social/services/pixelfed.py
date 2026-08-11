"""
Pixelfed API integration for Aiko
Handles media upload and status posting
"""

import requests
import os
from pathlib import Path
from typing import Optional, Dict, List
import json


class PixelfedAPI:
    def __init__(
        self,
        instance_url: str = None,
        access_token: str = None,
    ):
        """
        Initialize Pixelfed API client
        
        Args:
            instance_url: Your Pixelfed instance URL (e.g., https://pixelfed.social)
            access_token: Personal Access Token (can use env var PIXELFED_PAT)
        """
        self.instance_url = instance_url or os.getenv("PIXELFED_INSTANCE_URL", "https://pixelfed.social")
        self.access_token = access_token or os.getenv("PIXELFED_PAT")
        
        if not self.access_token:
            raise ValueError("PIXELFED_PAT environment variable not set")
        
        self.headers = {
            "Authorization": f"Bearer {self.access_token}",
            "Accept": "application/json",
        }
    
    def upload_media(
        self,
        file_path: str,
        description: str = "",
        filter_name: Optional[str] = None,
    ) -> Dict:
        """
        Upload media to Pixelfed
        
        Args:
            file_path: Path to image/video file
            description: Alt text / accessibility description
            filter_name: Optional filter name
            
        Returns:
            Media object with 'id' field for use in post creation
        """
        url = f"{self.instance_url}/api/v1/media"
        
        # Validate file exists
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        data = {}
        if description:
            data["description"] = description
        if filter_name:
            data["filter_name"] = filter_name
        
        with open(file_path, "rb") as f:
            files = {"file": (file_path.name, f)}
            response = requests.post(
                url,
                headers=self.headers,
                files=files,
                data=data,
            )
        
        if response.status_code not in [200, 201]:
            raise Exception(f"Media upload failed: {response.status_code} - {response.text}")
        
        return response.json()
    
    def upload_multiple_media(
        self,
        file_paths: List[str],
        descriptions: Optional[List[str]] = None,
    ) -> List[Dict]:
        """
        Upload multiple media files
        
        Args:
            file_paths: List of paths to media files
            descriptions: Optional list of alt texts (same length as file_paths)
            
        Returns:
            List of media objects
        """
        if descriptions and len(descriptions) != len(file_paths):
            raise ValueError("descriptions length must match file_paths length")
        
        media_list = []
        for i, file_path in enumerate(file_paths):
            desc = descriptions[i] if descriptions else ""
            media = self.upload_media(file_path, description=desc)
            media_list.append(media)
        
        return media_list
    
    def create_status(
        self,
        status: str,
        media_ids: Optional[List[str]] = None,
        sensitive: bool = False,
        spoiler_text: Optional[str] = None,
        visibility: str = "public",
        language: Optional[str] = None,
    ) -> Dict:
        """
        Create a status/post on Pixelfed
        
        Args:
            status: Post content (caption/text)
            media_ids: List of media IDs from upload_media()
            sensitive: Mark as sensitive/NSFW
            spoiler_text: Content warning text
            visibility: "public", "unlisted", "private", or "direct"
            language: ISO 639 language code (e.g., "en", "ja")
            
        Returns:
            Status object
        """
        url = f"{self.instance_url}/api/v1/statuses"
        
        data = {
            "status": status,
            "visibility": visibility,
        }
        
        if media_ids:
            data["media_ids"] = media_ids
        
        if sensitive:
            data["sensitive"] = True
        
        if spoiler_text:
            data["spoiler_text"] = spoiler_text
        
        if language:
            data["language"] = language
        
        response = requests.post(
            url,
            headers=self.headers,
            json=data,
        )
        
        if response.status_code not in [200, 201]:
            raise Exception(f"Status creation failed: {response.status_code} - {response.text}")
        
        return response.json()
    
    def post_with_image(
        self,
        caption: str,
        image_path: str,
        alt_text: str = "",
        sensitive: bool = False,
        visibility: str = "public",
    ) -> Dict:
        """
        Convenience method: upload image and create status in one call
        
        Args:
            caption: Post caption
            image_path: Path to image file
            alt_text: Accessibility description for image
            sensitive: Mark as NSFW
            visibility: Post visibility
            
        Returns:
            Status object
        """
        # Upload media
        media = self.upload_media(image_path, description=alt_text)
        media_id = media["id"]
        
        # Create status
        return self.create_status(
            status=caption,
            media_ids=[media_id],
            sensitive=sensitive,
            visibility=visibility,
        )
    
    def get_account_info(self) -> Dict:
        """Get current account information"""
        url = f"{self.instance_url}/api/v1/accounts/verify_credentials"
        response = requests.get(url, headers=self.headers)
        
        if response.status_code != 200:
            raise Exception(f"Failed to get account info: {response.status_code}")
        
        return response.json()


# Example usage for Aiko integration
if __name__ == "__main__":
    # Initialize
    pix = PixelfedAPI()
    
    # Verify account
    account = pix.get_account_info()
    print(f"Connected to account: @{account['username']}")
    
    # Simple post with image
    result = pix.post_with_image(
        caption="Just posted from Aiko! 📸",
        image_path="/path/to/image.jpg",
        alt_text="A photo description",
        visibility="public",
    )
    
    print(f"Posted! ID: {result['id']}")
    print(f"URL: {result['url']}")

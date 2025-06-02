import google.generativeai as genai
from PIL import Image
import os
from typing import Optional, Union, List

# Ensure GOOGLE_API_KEY is set or genai.configure() is called before use.
# Example:
# if os.getenv("GOOGLE_API_KEY"):
#     genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
# else:
#     print("Warning: GOOGLE_API_KEY not set. Token counting may fail or use default credentials.")


def count_gemini_tokens_simple(
    model_name: str,
    text_input: Optional[str] = None,
    image_input: Optional[Union[str, Image.Image]] = None
) -> int:
    """
    Counts the number of tokens for a given text and/or image input
    for a specified Gemini model (simplified version).

    Args:
        model_name (str): The name of the Gemini model (e.g., "gemini-1.5-flash-latest").
        text_input (Optional[str]): The text input.
        image_input (Optional[Union[str, Image.Image]]):
            The image input (file path or PIL.Image.Image object).

    Returns:
        int: The total number of tokens.
    """
    
    model = genai.GenerativeModel(model_name)
    parts: List[Union[str, Image.Image]] = []

    if text_input:
        parts.append(text_input)

    if image_input:
        if isinstance(image_input, str):
            img = Image.open(image_input)
            parts.append(img)
        elif isinstance(image_input, Image.Image):
            parts.append(image_input)
        # No explicit error for invalid image type in this simplified version

    if not parts:
        return 0 # No content, no tokens

    response = model.count_tokens(parts)
    return response.total_tokens


if __name__ == '__main__':
    # --- Example Usage ---
    # Ensure your GOOGLE_API_KEY is set in your environment,
    # or uncomment and use genai.configure() here.
    # For example:
    # api_key = os.getenv("GOOGLE_API_KEY")
    # if api_key:
    #     genai.configure(api_key=api_key)
    #     print(f"Using Gemini API Key: ...{api_key[-4:]}")
    # else:
    #     print("Warning: GOOGLE_API_KEY environment variable is not set for the example.")
    #     print("The example might fail or use default credentials if available.")

    # Configure API key for the example if it's set as an environment variable
    # This is important for the example to run.
    env_api_key = os.getenv("GOOGLE_API_KEY")
    if env_api_key:
        genai.configure(api_key=env_api_key)
        print(f"Configured Gemini API with key ending in: ...{env_api_key[-4:]}")
    else:
        print("Warning: GOOGLE_API_KEY not set. The __main__ example for token counting might not work.")
        print("Please set the GOOGLE_API_KEY environment variable.")

    if env_api_key: # Only run example if API key is likely configured
        test_model = "gemini-1.5-flash-latest" # Use a common, efficient model for testing

        # 1. Text only
        text_only_tokens = count_gemini_tokens_simple(
            model_name=test_model,
            text_input="This is a sample text for token counting."
        )
        print(f"Tokens for text only: {text_only_tokens}")

        # Create a dummy image for testing
        dummy_image_path = "dummy_for_token_count.png"
        try:
            img = Image.new('RGB', (60, 30), color = 'red')
            img.save(dummy_image_path)

            # 2. Image path only
            image_path_tokens = count_gemini_tokens_simple(
                model_name=test_model,
                image_input=dummy_image_path
            )
            print(f"Tokens for image (from path) only: {image_path_tokens}")

            # 3. PIL Image object only
            pil_image = Image.open(dummy_image_path)
            image_pil_tokens = count_gemini_tokens_simple(
                model_name=test_model,
                image_input=pil_image
            )
            print(f"Tokens for image (PIL object) only: {image_pil_tokens}")
            pil_image.close()

            # 4. Both text and image path
            text_and_image_tokens = count_gemini_tokens_simple(
                model_name=test_model,
                text_input="Describe this image.",
                image_input=dummy_image_path
            )
            print(f"Tokens for text and image (from path): {text_and_image_tokens}")

        except Exception as e:
            print(f"An error occurred during the __main__ example: {e}")
            print("This might be due to missing libraries (Pillow) or API key issues.")
        finally:
            if os.path.exists(dummy_image_path):
                os.remove(dummy_image_path)
    else:
        print("Skipping __main__ example execution as GOOGLE_API_KEY is not detected.")
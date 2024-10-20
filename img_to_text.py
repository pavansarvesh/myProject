import cv2
import easyocr
import google.generativeai as genai  # type: ignore
from IPython.display import Markdown
import os
import gc


def configure_genai():
    os.environ["GOOGLE_API_KEY"] = "AIzaSyDu-u4TKO92aM8yUSjCoiXM-WJV6v0ODYY"
    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

def api(total_text):
    model = genai.GenerativeModel('gemini-1.5-flash-8b')
    try:
        prompt = (
            f"{total_text} extract product name, flavor, usecase, MRP, Expiry date if present and ingredients from above in a tabular format. If not available, just extract text without throwing an error or showing an error message."
        )
        response = model.generate_content(prompt)
        return response.text
    finally:
        del model
        gc.collect()


# def to_markdown(text):
#     text = text.replace('.', '*')
#     return Markdown(textwrap.indent(text, '> ', predicate=lambda _: True))


def text_extractor(img_path):

    reader = easyocr.Reader(['en'])

    img = cv2.imread(img_path)

    results = reader.readtext(img)
    total_text = ' '.join([text[1] for text in results])

    del reader
    del img
    del results
    gc.collect()

    try:
        with open("detail.txt", "a+") as file:
            text_final = api(total_text)
            if text_final != "Unfortunately, the provided text is not in a standard format and is difficult to parse for the requested information.  The words appear to be a jumbled mix of product information, but critical details like product name, flavor, use case, MRP, expiry date, and ingredients are not discernible.":
                file.write(text_final)
    finally:

        del total_text
        del text_final
        gc.collect()

def main(img_path):
    configure_genai()
    text_extractor(img_path)
    # print(f"Text extraction and processing completed for {img_path}. Results saved to 'detail.txt'.")

    gc.collect()
main("./scrap/text.jpg")

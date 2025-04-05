# PDF Q&A Service

This is a Streamlit application that allows users to upload PDF documents and ask questions about their content. The application uses OpenAI's GPT-4 model to process the PDFs and provide accurate answers to user queries.

## Features

- PDF document upload and processing
- Interactive Q&A interface
- Conversation history tracking
- Vector-based document retrieval
- GPT-4 powered responses

## Installation

1. Clone this repository
2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```
3. Create a `.env` file in the project root and add your OpenAI API key:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```

## Usage

1. Run the Streamlit application:
   ```bash
   streamlit run app.py
   ```
2. Upload a PDF document using the file uploader
3. Wait for the document to be processed
4. Ask questions about the document's content
5. View the responses in the chat interface

## Requirements

- Python 3.8+
- OpenAI API key
- Required Python packages (listed in requirements.txt)

## Note

Make sure to replace `your_api_key_here` in the `.env` file with your actual OpenAI API key before running the application. 
# YouTube Video Summarizer

This project is a Flask web application that summarizes YouTube video transcripts using the LangChain library and a language model. The application takes a YouTube video URL as input and returns a concise summary of the video's main topic and key points.

## Features

- Validates YouTube URLs.
- Loads video transcripts using the `YoutubeLoader`.
- Summarizes the transcript using a `meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo`language model.
- Returns the summary in a user-friendly format via Twilio messaging.

## Technologies Used

- **Flask**: A lightweight WSGI web application framework for Python.
- **LangChain**: A framework for building applications with language models.
- **Twilio**: A cloud communications platform for sending and receiving messages.
- **Python**: The programming language used for the application.
- **dotenv**: For loading environment variables from a `.env` file.

## Concepts Used

- **Regular Expressions**: Utilized to validate YouTube URLs, ensuring that the input is a valid link to a YouTube video.
- **Document Loaders**: The `YoutubeLoader` is used to fetch video transcripts and metadata from YouTube.
- **Prompt Templates**: A structured way to format prompts for the language model, guiding it on how to process the input data (in this case, the video transcript).
- **LLM Chains**: Combines the language model and prompt template to create a processing chain that generates summaries based on the provided transcript.
- **Flask Routing**: Defines endpoints for the web application, allowing users to interact with the summarization functionality via HTTP requests.
- **Twilio Messaging**: Integrates with Twilio to send the generated summaries back to users in a messaging format.

## Future Scope and Planned Improvements

- **Multi-Lingual**: Support video transcripts in multiple languages.
- **More Summarization Options**: Implement different styles (e.g., bullet points, paragraphs) based on user preferences.
- **Integrate with custom video platforms**: Clients can integrate with their own video platforms.
- **Error Handling**: Current error handling is basic.Improve it to handle different types of errors e.g network errors,better default replies.
- **Frontend Interface**: Apart from whatsapp,we can have our own frontend interface for users to interact with.
- **Analytics Dashboard**: Client-defined engagement analytics dashboard to track usage statistics, popular video etc.

## Contributing

Contributions are welcome! If you have suggestions for improvements or new features,feel free to open an issue or submit a pull request. Please dont forget to give a star.

## Credits
- Credits to [LangChain](https://python.langchain.com/v0.1/docs/get_started/introduction/) and [Together](https://together.ai/) for providing the tools to build this. [Twilio](https://www.twilio.com/) for providing the Whatsapp API.
- My Teacher Harshit Tyagi for providing the guidance.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

import os
import uuid
import tiktoken
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from mysql.connector import connect, Error
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader

# Load environment variables
load_dotenv()

# Retrieve and validate OpenAI API key
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY is not set in the environment variables.")

# Retrieve and validate MySQL connection parameters
mysql_host = os.getenv("MYSQL_HOST", "localhost")
mysql_user = os.getenv("MYSQL_USER", "root")
mysql_password = os.getenv("MYSQL_PASSWORD", ".......")
mysql_database = os.getenv("MYSQL_DB", "hotel")

if not all([mysql_host, mysql_user, mysql_password, mysql_database]):
    raise ValueError("MySQL connection parameters are not properly set in the environment variables.")

# Initialize Flask app
app = Flask(__name__)

# Initialize OpenAI LLM
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# Load and process PDF document
pdf_path = "C:/Users/kamra\Desktop/newchat/hotel chatbot.pdf"  # Replace with the path to your PDF file
loader = PyPDFLoader(file_path=pdf_path)  # Correct parameter name
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
documents = loader.load_and_split(text_splitter=text_splitter)

# Create embeddings and FAISS vector store
embeddings = OpenAIEmbeddings()
vectordb = FAISS.from_documents(documents, embeddings)
retriever = vectordb.as_retriever()

# Initialize MySQL connection
def create_mysql_connection():
    try:
        connection = connect(
            host=mysql_host,
            user=mysql_user,
            password=mysql_password,
            database=mysql_database
        )
        print("Connected to MySQL successfully.")
        return connection
    except Error as e:
        raise ConnectionError(f"Failed to connect to MySQL: {e}")

# Chat endpoint
@app.route('/chat', methods=['POST'])
def chat():
    # Retrieve form data
    user_id = request.form.get('userID')
    session_id = request.form.get('sessionID')
    user_input = request.form.get('message')

    # Validate form data
    if not user_id or not user_input:
        return jsonify({'error': 'userID and message are required fields.'}), 400

    # Generate a new session_id if not provided
    if not session_id:
        session_id = str(uuid.uuid4())

    # Connect to MySQL
    connection = create_mysql_connection()
    cursor = connection.cursor()

    try:
        # Create chat history table if not exists
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chat_history (
                id INT AUTO_INCREMENT PRIMARY KEY,
                session_id VARCHAR(255),
                user_id VARCHAR(255),
                role VARCHAR(10),
                message TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Retrieve previous messages for the session
        cursor.execute("SELECT role, message FROM chat_history WHERE session_id = %s ORDER BY created_at", (session_id,))
        previous_messages = cursor.fetchall()

        # Format chat history for the prompt
        formatted_history = ""
        for role, message in previous_messages:
            formatted_history += f"{role}: {message}\n"

        # Add the current user input to the history
        formatted_history += f"User: {user_input}\n"

        # Retrieve relevant documents
        docs = retriever.get_relevant_documents(user_input)

        if docs:
            # If relevant documents are found, use them to generate a response
            context = " ".join([doc.page_content for doc in docs])
            prompt = f"Based on the following context and chat history, answer the question:\n\nContext: {context}\n\nChat History:\n{formatted_history}\nAI:"
        else:
            # If no relevant documents are found, use chat history only
            prompt = f"Based on the following chat history, answer the question:\n\nChat History:\n{formatted_history}\nAI:"

        # Generate AI response
        ai_response = llm.predict(prompt)

        # Add user and AI messages to history
        cursor.execute("INSERT INTO chat_history (session_id, user_id, role, message) VALUES (%s, %s, %s, %s)", 
                       (session_id, user_id, 'User', user_input))
        cursor.execute("INSERT INTO chat_history (session_id, user_id, role, message) VALUES (%s, %s, %s, %s)", 
                       (session_id, user_id, 'AI', ai_response))
        connection.commit()

        return jsonify({'response': ai_response, 'sessionID': session_id})

    except Error as e:
        return jsonify({'error': f"Database error: {str(e)}"}), 500

    finally:
        cursor.close()
        connection.close()

if __name__ == '__main__':
    app.run(debug=True)

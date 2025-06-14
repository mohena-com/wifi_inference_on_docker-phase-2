# CSV UI Application

This project provides a simple user interface for uploading a CSV file and sending the data to a Flask backend for prediction. The application allows users to interact with the machine learning model and receive predictions based on the uploaded data.

## Project Structure

```
csv-ui-app
├── index.html       # HTML structure for the user interface
├── script.js        # JavaScript code for handling file uploads and API requests
└── README.md        # Documentation for the project
```

## Files Description

- **index.html**: Contains the HTML structure for the user interface. It includes:
  - A file input for uploading a CSV file.
  - A button to submit the file.
  - A section to display the prediction results.

- **script.js**: Contains the JavaScript code that:
  - Handles the file upload.
  - Reads the CSV file and converts it to JSON format.
  - Sends a POST request to the `/predict` endpoint of the Flask application.
  - Processes the response and displays the results on the web page.

## Setup Instructions

1. **Clone the Repository**: 
   Clone this repository to your local machine using:
   ```
   git clone <repository-url>
   ```

2. **Navigate to the Project Directory**:
   ```
   cd csv-ui-app
   ```

3. **Run the Flask Application**:
   Ensure you have Flask installed and run the Flask application:
   ```
   python flask_real_time_inference.py
   ```

4. **Open the User Interface**:
   Open `index.html` in your web browser to access the user interface.

## Usage

1. Upload a CSV file using the file input.
2. Click the "Submit" button to send the data to the Flask backend.
3. View the prediction results displayed on the page.

## Requirements

- Flask
- JavaScript (for the frontend)
- HTML/CSS (for the user interface)

## License

This project is licensed under the MIT License.
let uploadedFeatures = []; // 1. Global variable



const csvToJson = (csv) => {
    const lines = csv.trim().split("\n").filter(line => line.trim() !== "");
    
    const csi_data = [];

    for (let i = 1; i < lines.length; i++) {
        const currentLine = lines[i].split(",");
        if (currentLine.length >= 2) {
            // Wrap each row (array) inside two extra arrays to get triple nesting
            csi_data.push([ currentLine.slice(0, -1)  ]);
        }
    }

    // Wrap the whole array inside another array to match triple nesting
    return { csi_data: [csi_data] };
};


const uploadFile = async () => { 

    const fileInput = document.getElementById("csvFile");
    const file = fileInput.files[0];
    console.log("Selected file:", file);
    if (!file) {
        alert("Please upload a CSV file.");
        return;
    }

    // Prepare FormData and append the file
    const formData = new FormData();
    formData.append("file", file);

    try {
        const response = // await fetch("/gaitid/predict", {
            await fetch("http://webmaster-ai:5002/gaitid/predict", {
            method: "POST",
            body: formData, // Send the file as form data
            // Do NOT set Content-Type header manually!
            // The browser will set the correct multipart/form-data headers automatically
        });

        console.log("Received response status:", response.status);
        const result = await response.json();
        console.log("Response JSON:", result);
        // displayResults(result);
    } catch (error) {
        console.error("Error while fetching:", error);
    }
};




const displayResults = (results) => {
    const resultsDiv = document.getElementById("results");
    resultsDiv.innerHTML = "";

    if (results.error) {
        resultsDiv.innerHTML = `<p>Error: ${results.error}</p>`;
        return;
    }

    results.forEach((result, index) => {
        const inputLine = uploadedFeatures[index] ? uploadedFeatures[index].join(", ") : "";
        const container = document.createElement("div");
        container.style.marginBottom = "1em";

        // Sober textbox for input
        const inputBox = document.createElement("textarea");
        inputBox.value = inputLine;
        inputBox.readOnly = true;
        inputBox.rows = 2;
        inputBox.style.width = "100%";
        inputBox.style.background = "#f5f5f5";
        inputBox.style.border = "1px solid #ccc";
        inputBox.style.borderRadius = "4px";
        inputBox.style.fontFamily = "monospace";
        inputBox.style.fontSize = "0.95em";
        inputBox.style.marginBottom = "0.5em";
        inputBox.style.resize = "none";

        const prediction = document.createElement("div");
        prediction.innerHTML = `<b>Predicted Activity:</b> ${result.predicted_activity}, <b>Confidence:</b> ${result.confidence}`;

        container.appendChild(inputBox);
        container.appendChild(prediction);
        resultsDiv.appendChild(container);
    });
};
//document.getElementById("uploadButton").onclick = uploadFile;

//document.getElementById("uploadButton").addEventListener("click", uploadFile);
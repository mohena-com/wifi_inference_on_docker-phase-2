let uploadedFeatures = []; // 1. Global variable

const csvToJson1 = (csv) => {
    const lines = csv.trim().split("\n").filter(line => line.trim() !== "");
    const headers = lines[0].split(",");
    const features = [];
    const csi_data = [];

    for (let i = 1; i < lines.length; i++) {
        const currentLine = lines[i].split(",");
        // Only push if the line has enough columns
        if (currentLine.length >= 2) {
            // Remove label (last column) for csi_data
            csi_data.push(currentLine.slice(0, -1).map(Number));
            // Keep the original features for display
            features.push(currentLine.slice(0, -1));
        }
    }

    return { csi_data, features };
};


const csvToJson = (csv) => {
    const lines = csv.trim().split("\n").filter(line => line.trim() !== "");
    
    const csi_data = [];

    for (let i = 1; i < lines.length; i++) {
        const currentLine = lines[i].split(",");
        if (currentLine.length >= 2) {
            // Wrap each row (array) inside two extra arrays to get triple nesting
            csi_data.push([ currentLine.slice(0, -1).map(Number) ]);
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

    const reader = new FileReader();
    reader.onload = async (event) => {
        const csv = event.target.result;
        console.log("csv:", csv);
        const jsonData = csvToJson(csv);
        console.log("jsonData:", jsonData);
        uploadedFeatures = jsonData.features; // 3. Store features for display
        console.log("Uploaded Features:", uploadedFeatures);    
    
        try {
            console.log("Sending POST request with CSI data:", JSON.stringify({ csi_data: jsonData.csi_data }));
            const response = await fetch("/gaitid/predict", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                },
                body: JSON.stringify({ csi_data: jsonData.csi_data }),
            });
        console.log("Received response status:", response.status);
        const result = await response.json();
        console.log("Response JSON:", result);
        displayResults(result);
        // further processing...
        } catch (error) {
            console.error("Error while fetching:", error);
        }

    };

    reader.readAsText(file);
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

document.getElementById("uploadButton").addEventListener("click", uploadFile);
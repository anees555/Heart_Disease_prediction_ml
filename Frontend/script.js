document.getElementById('heartForm').addEventListener('submit', async function (e) {
    e.preventDefault(); // Prevent page reload
    
    // Gather form data
    const formData = new FormData(e.target);
    const data = {};
    formData.forEach((value, key) => (data[key] = parseFloat(value) || value));

    // Call the backend API
    try {
        const response = await fetch('http://127.0.0.1:8000/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data),
        });

        const result = await response.json();
        const prediction = result.prediction === 1
            ? "You have heart disease"
            : "You do not have heart disease";

        // Store the result in localStorage and redirect
        localStorage.setItem('prediction', prediction);
        window.location.href = 'result.html';
    } catch (error) {
        console.error('Error:', error);
        alert('An error occurred while processing your request.');
    }
});

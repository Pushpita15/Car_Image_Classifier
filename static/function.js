const chatOutput = document.getElementById('chat-output');
const userInput = document.getElementById('user-input');
const predictionsOutput = document.getElementById('predictions-output');
const imageForm = document.getElementById('image-form');

imageForm.addEventListener('submit', function (event) {
    event.preventDefault(); // Prevent the default form submission behavior

    const file = imageForm.querySelector('#form-image-upload').files[0];


    if (!file) return;

    // Send the image file to the backend for predictions
    const formData = new FormData();
    formData.append('image', file);

    fetch('/predict', {
        method: 'POST',
        body: formData,
    })
    .then(response => response.json())
    .then(data => {
        if (data.predictions) {
            // Display predictions in the predictionsOutput div
            //predictionsOutput.innerText = `Predictions: ${data.predictions.join(', ')}`;
            appendImage(file, data.predictions.join(', '), 'user');
             // Display car data in the chatOutput div
            appendCarData(data.car_data);
        } else {
            predictionsOutput.innerText = 'No predictions available.';
        }
    })
    .catch(error => console.error('Error:', error));

    
});



function appendImage(imageFile, predictions, sender) {

    // Clear previous content in chatOutput
    chatOutput.innerHTML = '';

    
    const imageDiv = document.createElement('div');
    imageDiv.className = sender === 'user' ? 'user-message' : 'ai-message';
    
    // Display predictions in the imageDiv
    const predictionsDiv = document.createElement('div');
    predictionsDiv.innerText = `Predictions: ${predictions}`;
    imageDiv.appendChild(predictionsDiv);

    // Display the image in the imageDiv
    const img = document.createElement('img');
    const reader = new FileReader();
    reader.onload = function () {
        img.src = reader.result;
        img.style.maxWidth = '50%';
        img.style.height = 'auto';
        imageDiv.appendChild(img);
        chatOutput.appendChild(imageDiv);
        chatOutput.scrollTop = chatOutput.scrollHeight;
    };
    reader.readAsDataURL(imageFile);
}


function appendCarData(carData) 
{
    console.log('Type of carData:', typeof carData);
    const carDataDiv = document.createElement('div');
    carDataDiv.className = 'ai-message';
    
    carData.forEach(car => {
        const carInfo = document.createElement('p');
        carInfo.innerText = `Model: ${car.Model}, Year: ${car.Year}, Fuel: ${car.Fuel}, Price: ${car.Price},Transmission: ${car.Transmission}`;
        carDataDiv.appendChild(carInfo);
    });

    chatOutput.appendChild(carDataDiv);
    chatOutput.scrollTop = chatOutput.scrollHeight;
}



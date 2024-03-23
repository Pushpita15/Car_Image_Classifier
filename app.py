from flask import Flask, render_template, request, jsonify
import torch
from PIL import Image
from torchvision import transforms
import pymysql
from ml_module import load_model, process_image, predict , load_class_mapping 
import io
app = Flask(__name__)

# Loading your machine learning model
model = load_model()  


# Establishing a connection
connection = pymysql.connect(
    host='127.0.0.1',
    user='root',
    password='1234',
    database='car',
    cursorclass=pymysql.cursors.DictCursor  
)
predictions = ""
@app.route('/predict', methods=['POST'])
def predict_route():
    global predictions
    try:
        # Get the image file from the request
        file = request.files['image']

        # Process the image
        image = Image.open(io.BytesIO(file.read()))
        input_tensor = process_image(image)

        # Make predictions using your predict function
        class_mapping = load_class_mapping("D:\car_classification_webapp\idx_to_class.json")
        predictions = predict(input_tensor,model,class_mapping)  # Implement this function in your_ml_module

        # Return predictions as JSON
        print(predictions[0])

         # Retrieve data from the MySQL database based on predictions
        car_data = get_car_data(predictions[0])
        #print(type(car_data))
        #print(car_data)
        # Prepare the response
        if len(car_data) > 0:
            response = {'predictions': predictions, 'car_data': car_data}
        else:
            response = {'predictions': predictions, 'car_data': 'Sorry!! WE ARE NOT HAVING THE DATA YOU ARE SEARCHING FOR'}

        # Return predictions and car data as JSON
        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e)})
    
# Function to get car data from the MySQL database
def get_car_data(predictions):
    try:
        

        
        
        car_names = predictions.split(' ')
        if(len(car_names)>1):
            # Use the connection
            with connection.cursor() as cursor:
                # Execute SQL query
                cursor.execute("SELECT * FROM car_details WHERE company LIKE %s and model LIKE %s", ('%' + car_names[0] + '%','%' + car_names[1] + '%'))
                # Fetch results
                car_data = cursor.fetchall()
        else: 
            cursor.execute("SELECT * FROM car_details WHERE model LIKE %s", ('%' + predictions + '%',))
                # Fetch results
            car_data = cursor.fetchall()
            
            
        
        # Use the connection
        #with connection.cursor() as cursor:
            # Execute SQL query
            #cursor.execute("SELECT * FROM car_details WHERE company LIKE %s and model LIKE %s", ('%' + car_names[0] + '%','%' + car_names[1] + '%'))
            # Fetch results
            #car_data = cursor.fetchall()
            #if(len(car_data)==0):
                # Execute SQL query
                #cursor.execute("SELECT * FROM car_details WHERE company LIKE %s", ('%' + car_names[1] + '%',))
                # Fetch results
                #car_data = cursor.fetchall()
                
            

        # Close the connection
        #connection.close()
        return car_data

    except Exception as e:
        return str(e)
    
    
@app.route('/', methods=['GET'])
def homepage():

    return render_template("struct.html")

if __name__ == '__main__':
    app.run(debug=True)

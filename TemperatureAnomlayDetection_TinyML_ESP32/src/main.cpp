#include <Arduino.h>

// Include the TensorFlow Lite library
#include <TensorFlowLite_ESP32.h>
#include "tensorflow/lite/micro/kernels/micro_ops.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"

// Include the model data (this is the autoencoder model trained in Python)
#include "temperature_model.h"

// Include the DHT library for the DHT sensor
#include "DHTesp.h"
#include <LiquidCrystal_I2C.h>

/** Configuration for the TensorFlow Lite model  **/

const int kInputSize = 5;  // Number of input features
const int kOutputSize = 5; // Number of output values (same as input for autoencoder)

// Parameters for scaling (replace these with the actual values used during training)
const float mean[5] = {17.65814788, 24.08938664, 32.08945356, 6.53680408, 1925.7543933};
const float standard_deviation[5] = {9.91537675, 22.92565036, 73.91146513, 3.40136724, 53.52918998};

// Threshold for anomaly detection (adjust as needed)
const float threshold = 300;

uint8_t tensor_arena[10 * 1024]; // Adjust the size as necessary

tflite::MicroInterpreter *interpreter = nullptr;
tflite::ErrorReporter *error_reporter = nullptr;
tflite::MicroMutableOpResolver<10> resolver;
tflite::MicroAllocator *allocator = nullptr;

/** Configuration for DHT Sensor **/

#define I2C_ADDR 0x27
#define LCD_COLUMNS 16
#define LCD_LINES 2

const int DHT_PIN = 15;

DHTesp dhtSensor;

LiquidCrystal_I2C lcd(I2C_ADDR, LCD_COLUMNS, LCD_LINES);

void setup()
{
    Serial.begin(115200);
    Serial.println("Starting...");

    // Setup DHT sensor and LCD
    dhtSensor.setup(DHT_PIN, DHTesp::DHT22);
    lcd.init();
    lcd.backlight();

    // Check available heap memory
    Serial.print("Available heap memory: ");
    Serial.println(heap_caps_get_free_size(MALLOC_CAP_8BIT));

    // Setup TensorFlow Lite model
    error_reporter = tflite::GetMicroErrorReporter();
    const tflite::Model *model = tflite::GetModel(temperature_model);
    if (model->version() != TFLITE_SCHEMA_VERSION)
    {
        Serial.println("Model provided is schema version doesn't match!");
        return;
    }

    tflite::AllOpsResolver resolver;
    static tflite::MicroInterpreter static_interpreter(model, resolver, tensor_arena, sizeof(tensor_arena), error_reporter);
    interpreter = &static_interpreter;

    if (interpreter->AllocateTensors() != kTfLiteOk)
    {
        Serial.println("AllocateTensors() failed");
        return;
    }

    Serial.println("Setup done.");
}

void scaleInput(float *input)
{
    for (int i = 0; i < kInputSize; i++)
    {
        input[i] = (input[i] - mean[i]) / standard_deviation[i];
    }
}

void unscaleOutput(float *output)
{
    for (int i = 0; i < kOutputSize; i++)
    {
        output[i] = output[i] * standard_deviation[i] + mean[i];
    }
}

void loop()
{
    TempAndHumidity data = dhtSensor.getTempAndHumidity();

    float input_data[kInputSize] = {data.temperature, 5.63, -75.87, 6, 2024};

    // Print input data before scaling
    Serial.println("/-----------------------------------/");

    Serial.println("Input (Before scaling):");
    for (int i = 0; i < kInputSize; i++)
    {
        Serial.print(input_data[i]);
        Serial.print(" ");
    }
    Serial.println();

    // Scale the input data
    scaleInput(input_data);

    // Fill the input tensor
    float *input = interpreter->typed_input_tensor<float>(0);
    for (int i = 0; i < kInputSize; i++)
    {
        input[i] = input_data[i];
    }

    Serial.println("/-----------------------------------/");

    // Print input data
    Serial.println("Input (After scaling):");
    for (int i = 0; i < kInputSize; i++)
    {
        Serial.print(input[i]);
        Serial.print(" ");
    }
    Serial.println();

    // Run inference
    if (interpreter->Invoke() != kTfLiteOk)
    {
        Serial.println("Invoke failed!");
        return;
    }

    // Get the output from the model
    float *output = interpreter->typed_output_tensor<float>(0);

    // Calculate Mean Squared Error (MSE)
    float mse = 0.0;
    for (int i = 0; i < kOutputSize; i++)
    {
        mse += pow(input[i] - output[i], 2);
    }
    mse /= kOutputSize;

    // Identify anomalies
    bool is_anomaly = mse > threshold;

    Serial.println("/-----------------------------------/");

    // Print result
    Serial.print("MSE: ");
    Serial.println(mse);
    Serial.print("Anomaly Detected: ");
    Serial.println(is_anomaly ? "Yes" : "No");

    lcd.setCursor(0, 0);
    lcd.print("  Temp: " + String(data.temperature, 1) + "\xDF" + "C  ");
    lcd.setCursor(0, 1);
    lcd.print(is_anomaly ? "Anomaly detected" : "Normal behavior");

    delay(1000); // Wait for a second before the next prediction
}
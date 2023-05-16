
package com.example.tensorflow_test;

import static android.Manifest.permission.RECORD_AUDIO;

import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;

import android.graphics.Typeface;
import android.speech.RecognitionListener;
import android.speech.RecognizerIntent;
import android.speech.SpeechRecognizer;
import android.speech.tts.TextToSpeech;
import android.Manifest;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.media.ThumbnailUtils;
import android.os.Bundle;
import android.provider.MediaStore;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;

import com.example.tensorflow_test.ml.ModelUnquant;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.Locale;

public class MainActivity extends AppCompatActivity {

    TextToSpeech outputSpeech;
    TextView result;
    ImageView imageView;
    Button picture, micButton;
    int imageSize = 224;
    SpeechRecognizer speechRecognizer;
    Intent intentRecognizer;
    Float confidence;





    @Override
    protected void onCreate(Bundle savedInstanceState) {

        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        ActivityCompat.requestPermissions(this, new String[]{RECORD_AUDIO}, PackageManager.PERMISSION_GRANTED);

        intentRecognizer = new Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH);
        intentRecognizer.putExtra(RecognizerIntent.EXTRA_LANGUAGE_MODEL, RecognizerIntent.LANGUAGE_MODEL_FREE_FORM);

        speechRecognizer = SpeechRecognizer.createSpeechRecognizer(this);
        speechRecognizer.setRecognitionListener(new RecognitionListener() {
            @Override
            public void onReadyForSpeech(Bundle params) {

            }

            @Override
            public void onBeginningOfSpeech() {

            }

            @Override
            public void onRmsChanged(float rmsdB) {

            }

            @Override
            public void onBufferReceived(byte[] buffer) {

            }

            @Override
            public void onEndOfSpeech() {

            }

            @Override
            public void onError(int error) {

            }

            @Override
            public void onResults(Bundle results) {
                ArrayList<String> matches = results.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION);
                if(matches!= null) {
                    System.out.println(matches);
                    if (matches.contains("picture"))
                        if (checkSelfPermission(Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
                            Intent cameraIntent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
                            startActivityForResult(cameraIntent, 1);
                        } else {
                            //Request camera permission if we don't have it.
                            requestPermissions(new String[]{Manifest.permission.CAMERA}, 100);
                        }
                    else if(matches.contains("repeat")){
                        ConvertTextToSpeech();
                    }
                    else if(matches.contains("confidence")){
                        if(result.getText() != "Unsure, please retake picture.") {
                            outputSpeech.speak(String.valueOf(Math.floor(confidence * 100 * 10) / 10) + " percent confident.", TextToSpeech.QUEUE_FLUSH, null);
                        }
                    }
                    else if(matches.contains("help")){
                        String helpText = "Bottom left is the take picture button. Bottom right is the mic button. Take" +
                                " picture will send you to your phones camera to take a picture. Having the bill be flat and visible will ensure the best results";
                        outputSpeech.speak(helpText, TextToSpeech.QUEUE_FLUSH, null);
                    }
                }


            }

            @Override
            public void onPartialResults(Bundle partialResults) {

            }

            @Override
            public void onEvent(int eventType, Bundle params) {

            }
        });

        outputSpeech=new TextToSpeech(MainActivity.this, new TextToSpeech.OnInitListener() {


            @Override
            public void onInit(int status) {
                // TODO Auto-generated method stub
                if(status == TextToSpeech.SUCCESS){
                    int result=outputSpeech.setLanguage(Locale.US);
                    outputSpeech.setSpeechRate(0.65f);
                    if(result==TextToSpeech.LANG_MISSING_DATA ||
                            result==TextToSpeech.LANG_NOT_SUPPORTED){
                    }
                    else{
                        ConvertTextToSpeech();
                    }
                }
            }
        });

        result = findViewById(R.id.result);
        imageView = findViewById(R.id.imageView);
        picture = findViewById(R.id.button);
        micButton = findViewById(R.id.btn_speak);

        picture.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                // Launch camera if we have permission
                if (checkSelfPermission(Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
                    Intent cameraIntent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
                    startActivityForResult(cameraIntent, 1);
                } else {
                    //Request camera permission if we don't have it.
                    requestPermissions(new String[]{Manifest.permission.CAMERA}, 100);
                }
            }
        });

        micButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                speechRecognizer.startListening(intentRecognizer);
            }
        });
    }

    public void classifyImage(Bitmap image){
        try {
            ModelUnquant model = ModelUnquant.newInstance(getApplicationContext());

            // Creates inputs for reference.
            TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1, 224, 224, 3}, DataType.FLOAT32);


            ByteBuffer byteBuffer = ByteBuffer.allocateDirect(4* imageSize * imageSize *3);
            byteBuffer.order(ByteOrder.nativeOrder());

            int [] intValues = new int[imageSize * imageSize];
            image.getPixels(intValues, 0, image.getWidth(), 0, 0, image.getWidth(), image.getHeight());
            int pixel = 0;
            for(int i = 0; i < imageSize; i++)
                for(int j = 0; j <imageSize; j++){
                    int val = intValues[pixel++];
                    byteBuffer.putFloat(((val >> 16) & 0xFF) * (1.f / 255.f));
                    byteBuffer.putFloat(((val >> 8) & 0xFF) * (1.f / 255.f));
                    byteBuffer.putFloat((val & 0xFF) * (1.f / 255.f));
                }

            inputFeature0.loadBuffer(byteBuffer);
            // Runs model inference and gets result.
            ModelUnquant.Outputs outputs = model.process(inputFeature0);

            TensorBuffer outputFeature0 = outputs.getOutputFeature0AsTensorBuffer();

            float[] confidences = outputFeature0.getFloatArray();



            //System.out.println(confidences.length);
            int maxPos = 0;
            float maxConfidence = 0;
            for(int i = 0; i <confidences.length; i++){
                System.out.println(confidences[i]*100);
                if(confidences[i] > maxConfidence){
                    maxConfidence = confidences[i];
                    maxPos = i;
                }
            }
            confidence = maxConfidence;
            String[] classes = {"One", "Five", "Ten", "Twenty", "Fifty", "One Hundred"};


/*
            for(int i =0; i< classes.length; i++){
                System.out.println(classes[i]+","+confidences[i]*100);
            }
*/
            result.setText(classes[maxPos]);

            if(maxConfidence <= .90f){
                result.setText("Unsure, please retake picture.");
            }
            Typeface customFont = Typeface.createFromAsset(getAssets(), "fonts/Inter-Black.ttf");
            result.setTypeface(customFont);
            ConvertTextToSpeech();



            // Releases model resources if no longer used.
            model.close();
        } catch (IOException e) {
            // TODO Handle the exception
        }
    }

    @Override
    public void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        if (requestCode == 1 && resultCode == RESULT_OK) {
            Bitmap image = (Bitmap) data.getExtras().get("data");
            int dimension = Math.min(image.getWidth(), image.getHeight());
            //System.out.println(image.getWidth() + " " +image.getHeight());
            image = ThumbnailUtils.extractThumbnail(image, dimension, dimension);
            imageView.setImageBitmap(image);

            image = Bitmap.createScaledBitmap(image, imageSize, imageSize, true);
            classifyImage(image);
        }
        super.onActivityResult(requestCode, resultCode, data);
    }

    private void ConvertTextToSpeech() {
        // TODO Auto-generated method stub
        String text = result.getText().toString();
        if(text==null||"".equals(text))
        {
            text = "";
            outputSpeech.speak(text, TextToSpeech.QUEUE_FLUSH, null);
        }else if("One".equals(text) || "Two".equals(text) || "Five".equals(text) || "Ten".equals(text) || "Twenty".equals(text) || "Fifty".equals(text) || "One Hundred".equals(text) )
            outputSpeech.speak(text+" U S D", TextToSpeech.QUEUE_FLUSH, null);
        else
            outputSpeech.speak(text, TextToSpeech.QUEUE_FLUSH, null);
    }

    private void changeCurrency(){

    }


}
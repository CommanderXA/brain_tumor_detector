import React, { useEffect, useRef, useState } from "react";
import axios from "axios";

import SecondaryButton from "../components/buttons/button";
import ImageViewer from "../components/image_viewer/image_viewer";
import styles from "./home.module.scss";

interface ImageFile {
  image: string | null;
  filename: string | null;
}

interface Prediction {
  prediction: string;
  certainty: number | null;
}

const Home = () => {
  const [imageFile, setImageFile] = useState<ImageFile>({
    image: null,
    filename: null,
  });
  const [prediction, setPredicition] = useState<Prediction>({
    prediction: "Prediction".toUpperCase(),
    certainty: null,
  });
  const uploadImageRef = useRef<any>();

  const convertToBase64 = (file: any) => {
    return new Promise((resolve, reject) => {
      const fileReader = new FileReader();
      fileReader.readAsDataURL(file);
      fileReader.onload = () => {
        resolve(fileReader.result);
      };
      fileReader.onerror = (error) => {
        reject(error);
      };
    });
  };

  const handleChange = async (e: any) => {
    const [file] = e.target.files;
    if (file && file.type.substring(0, 5) === "image") {
      const image = String(await convertToBase64(file));
      setImageFile({ image: image, filename: file.name });
    } else {
      alert("Please, choose the image file!");
    }
  };

  const handleRemove = () => {
    setImageFile({ image: null, filename: null });
    setPredicition({ prediction: "Prediction".toUpperCase(), certainty: null });
    console.log(
      prediction.prediction.toLowerCase() == "healthy"
        ? "#3ab665"
        : prediction.prediction.toLowerCase() == "tumor"
        ? "#b63a3a"
        : "#00000"
    );
  };

  const getPrediction = async () => {
    const result = await axios.post(
      "http://127.0.0.1:8081/predict/image",
      JSON.stringify({
        image: imageFile.image,
        filename: imageFile.filename,
      }),
      { headers: { "Content-Type": "application/json" } }
    );

    setPredicition({
      prediction: result.data.prediction.class,
      certainty: result.data.prediction.certainty,
    });
  };

  return (
    <div className={styles.center}>
      <div className={styles.container}>
        <div style={{ display: "flex", gap: "16px" }}>
          <h1
            style={{
              margin: 0,
              padding: 0,
              fontSize: 42,
              fontWeight: 800,
              color:
                prediction.prediction.toLowerCase() == "normal"
                  ? "#3ab665"
                  : prediction.prediction.toLowerCase() == "tumor"
                  ? "#b63a3a"
                  : "#00000",
            }}
          >
            {prediction.prediction.toUpperCase()}
          </h1>
          <h1 style={{ margin: 0, padding: 0, fontSize: 42, fontWeight: 600 }}>
            {prediction.certainty !== null ? Math.floor(prediction.certainty) + "%" : ""}
          </h1>
        </div>
        <ImageViewer image={imageFile.image} />

        <div style={{ display: "flex", gap: 16 }}>
          {/* Upload Button */}
          <SecondaryButton
            variant={2}
            fit={true}
            onClick={() =>
              imageFile.image === null
                ? uploadImageRef.current.click()
                : handleRemove()
            }
          >
            {imageFile.image === null ? "Upload" : "Remove"}
          </SecondaryButton>

          {/* Check */}
          <SecondaryButton
            variant={1}
            fit={false}
            disabled={imageFile.image === null ? true : false}
            onClick={() => getPrediction()}
          >
            {"Check"}
          </SecondaryButton>
        </div>
      </div>

      {/* Hidden file input */}
      <input
        type="file"
        accept="image/*"
        id="image_input"
        hidden
        multiple={false}
        ref={uploadImageRef}
        onChange={handleChange}
      />
    </div>
  );
};

export default Home;

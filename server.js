import dotenv from "dotenv";
dotenv.config();
import express from "express";
import cors from "cors";
import fileUpload from "express-fileupload";
// import { upload } from "./middlewares/multer.js";
import * as tf from "@tensorflow/tfjs-node";
import { readFile } from "fs/promises";
// import predictionModel from "./utility/location.keras";

// load model
const model = await tf.loadLayersModel("file://utility/model/model.json");
const labels = [
  "Brinjal",
  "Carrot",
  "Chilli",
  "Garlic",
  "Ginger",
  "Onion",
  "Potato",
  "Tomato",
];

const app = express();

app.use(cors());
// app.use(fileUpload());
app.use(
  fileUpload({
    useTempFiles: true,
    tempFileDir: "/tmp/",
  })
);

app.use(express.json());
app.use(express.urlencoded({ extended: true }));

app.get("/", (req, res) => {
  res.json({ data: "hello world" });
});

// app.post("/upload", predictResult);

// function predictResult(req, res){

// }

app.post("/upload", async (req, res) => {
  // console.log(req.files.image);
  const imageBuffer = await readFile(req.files.image.tempFilePath);
  // console.log(imageBuffer);
  let image = tf.node.decodeImage(imageBuffer);

  image = tf.image.resizeNearestNeighbor(image, [128, 128]).expandDims();

  const predictions = await model.predict(image).data();
  // console.log(predictions);
  const result = labels[predictions.indexOf(Math.max(...predictions))];
  console.log(result);
  // console.log(req.files.image);

  // const predictions = await model.predict(image).data();

  return res.json({ status: "ok", data: result });
});

app.listen(process.env.PORT, () => {
  console.log("server is running at port: ", process.env.PORT);
});

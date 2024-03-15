import dotenv from "dotenv";
dotenv.config();
import express from "express";
import cors from "cors";
import fileUpload from "express-fileupload";
// import { upload } from "./middlewares/multer.js";
import * as tf from "@tensorflow/tfjs-node";
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
app.use(fileUpload());

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
  let image = tf.node.decodeImage(req.files.image.data);
  // let image = tf.node.decodeImage(req.body);
  image = tf.image.resizeNearestNeighbor(image, [128, 128]).expandDims();
  // const image = req.body.image;
  // tf.resize
  // console.log(image.dataSync());
  const predictions = await model.predict(image).data();
  console.log(predictions);
  const result = labels[predictions.indexOf(Math.max(...predictions))];
  console.log(result);
  // console.log(req.files.image);

  // const predictions = await model.predict(image).data();

  return res.json({ status: "ok", data: result });
});

app.listen(process.env.PORT, () => {
  console.log("server is running at port: ", process.env.PORT);
});

// testServer.js
const express = require("express");
const app = express();
app.use(express.json());

app.post("/processFile", (req, res) => {
  console.log("Received at backend:", req.body);
  res.json({ status: "got it" });
});

app.listen(3333, () => console.log("Fake backend on 3333"));

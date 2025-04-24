const express = require('express');
const { execFile } = require('child_process');
const path = require('path');
const app = express();
const port = 3000;

app.use(express.json());
app.use(express.static(path.join(__dirname, 'public')));  // 🧠 Serves index.html

app.post('/predict', (req, res) => {
  const inputData = req.body.input;

  execFile('python', ['predict.py', JSON.stringify(inputData)], (error, stdout, stderr) => {
    if (error) {
      console.error('Prediction error:', error);
      return res.status(500).json({ error: 'Prediction failed' });
    }
    res.json({ prediction: parseInt(stdout.trim()) });
    console.log({ prediction: parseInt(stdout.trim()) })
  });
});

app.listen(port, () => {
  console.log(`🌐 Web server running at http://localhost:${port}`);
});

import { useState } from "react";
import './App.css';

function App() {
  const [inputText, setInputText] = useState("");
  const [outputText, setOutputText] = useState("");

  const handleGenerate = async () => {
    const res = await fetch("http://localhost:8000/generate", {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({ input: inputText })
    });

    const data = await res.json();
    setOutputText(data.output);
  };

  return (
    <div className="App">
      <h1>Flan-T5 Text Generator</h1>
      <textarea
        rows="4"
        cols="50"
        value={inputText}
        onChange={(e) => setInputText(e.target.value)}
        placeholder="Type your prompt here"
      />
      <br />
      <button onClick={handleGenerate}>Generate</button>
      <h3>Output:</h3>
      <p>{outputText}</p>
    </div>
  );
}

export default App;

#  Cyberbullying Detection Component

##  Overview

This component provides **real-time cyberbullying detection** for social media platforms.
It analyzes **text comments** and **meme content (image + caption)** and returns a moderation decision.

The system is designed as a **platform-independent AI service**, which can be integrated into any application via HTTP APIs.

---

##  Features

* Detects toxic and abusive comments
* Analyzes memes using image + text
* Hybrid approach (AI models + keyword detection)
* Returns severity and action (ALLOW / WARN / DELETE)

---

##  Project Structure

```text
components/cyberbullying/
├── app/         # FastAPI endpoints
├── core/        # Inference logic
├── models/      # Model files (not included in Git)
├── assets/      # Toxic & severe word lists
└── README.md
```

---

##  Running Locally (For Development)

```bash
uvicorn components.cyberbullying.app.main:app --reload
```

Open in browser:

```
http://127.0.0.1:8000/docs
```

---

##  API Endpoints

### POST `/moderate_comment`

```json
{ "text": "your comment here" }
```

---

### POST `/moderate_meme`

Form-data:

* `image` (file)
* `caption` (text)

---

##  Integration (PHP Example)

To integrate this system into a PHP-based platform (e.g., OSSN), create a file named:

```
moderation_api.php
```

Add the following code:

```php
<?php

function moderate_comment_api($comment_text){

    $url = "http://127.0.0.1:8000/moderate_comment"; // Replace with deployed API URL

    $payload = json_encode([
        'text' => $comment_text
    ]);

    $ch = curl_init();

    curl_setopt($ch, CURLOPT_URL, $url);
    curl_setopt($ch, CURLOPT_POST, true);
    curl_setopt($ch, CURLOPT_POSTFIELDS, $payload);
    curl_setopt($ch, CURLOPT_RETURNTRANSFER, true);
    curl_setopt($ch, CURLOPT_HTTPHEADER, [
        'Content-Type: application/json'
    ]);

    $response = curl_exec($ch);
    curl_close($ch);

    return json_decode($response, true);
}

?>
```

### Usage Example

```php
$result = moderate_comment_api("You are stupid");
print_r($result);
```

---

##  Notes

* Model files are **not included** due to size limits
* Place trained models inside the `models/` folder for local testing
* When deployed, replace `http://127.0.0.1:8000` with your **live API URL**

---

## 👨 Author
Nesha Wickremasinghe
Cyberbullying Component – AI Plugin Project

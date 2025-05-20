# Rainfall Prediction Web Application

A machine learning-based web application that predicts rainfall probability based on weather parameters.

## Features

- Real-time rainfall prediction
- Location-based weather data
- Historical prediction tracking
- Weather trend visualization
- Mobile-responsive design

## Tech Stack

- Python 3.11
- Flask
- SQLite
- scikit-learn
- Tailwind CSS
- Chart.js

## Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/rainfall-prediction.git
cd rainfall-prediction
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
Create a `.env` file with:
```
OPENWEATHER_API_KEY=your_api_key_here
FLASK_ENV=development
```

4. Run the application:
```bash
python app.py
```

5. Open your browser and go to `http://localhost:5000`

## Project Structure

- `app.py`: Main Flask application
- `database.py`: SQLite database operations
- `templates/`: HTML templates
- `models/`: Trained ML models
- `static/`: Static assets

## API Endpoints

- `GET /`: Home page
- `POST /predict`: Make rainfall prediction
- `GET /history`: Get prediction history
- `GET /locations`: Get saved locations
- `POST /add_location`: Add new location

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

MIT License 
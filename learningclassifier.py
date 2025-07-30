import google.generativeai as genai
import numpy as np
import sys
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import plotext as pltxt
import re
import json
import random

API_KEY = "api key"
genai.configure(api_key=API_KEY)

questions = [
    {
        "question": "How do you best learn new technical concepts in BTech courses?",
        "options": [
            ("Watching video tutorials or circuit diagrams", "Visual"),
            ("Listening to professor lectures or podcasts", "Auditory"),
            ("Reading textbooks or writing code comments", "Reading/Writing"),
            ("Building circuits or coding projects", "Kinesthetic"),
            ("Deriving equations or debugging code", "Logical/Analytical"),
            ("Group study or project discussions", "Social/Interpersonal"),
            ("Self-study with coding practice", "Solitary/Intrapersonal"),
            ("Relating concepts to real-world systems", "Naturalist")
        ]
    },
    {
        "question": "What helps you retain complex engineering concepts?",
        "options": [
            ("Flowcharts or simulation visuals", "Visual"),
            ("Technical discussions or audio explanations", "Auditory"),
            ("Writing summaries or reading journals", "Reading/Writing"),
            ("Hands-on labs or prototype building", "Kinesthetic"),
            ("Logical analysis or problem-solving", "Logical/Analytical"),
            ("Collaborating on team projects", "Social/Interpersonal"),
            ("Independent research or reflection", "Solitary/Intrapersonal"),
            ("Observing real-world applications", "Naturalist")
        ]
    },
    {
        "question": "Which BTech classroom activity engages you most?",
        "options": [
            ("Watching software demos or schematics", "Visual"),
            ("Listening to case studies or debates", "Auditory"),
            ("Solving written problems or coding", "Reading/Writing"),
            ("Lab experiments or hardware tasks", "Kinesthetic"),
            ("Algorithm design or logical proofs", "Logical/Analytical"),
            ("Team-based coding or projects", "Social/Interpersonal"),
            ("Solo assignments or coding", "Solitary/Intrapersonal"),
            ("Field visits to tech industries", "Naturalist")
        ]
    },
    {
        "question": "How do you prepare for BTech exams or projects?",
        "options": [
            ("Using diagrams or mind maps", "Visual"),
            ("Listening to recorded lectures", "Auditory"),
            ("Reading notes or technical papers", "Reading/Writing"),
            ("Practicing with hardware or code", "Kinesthetic"),
            ("Breaking down problems logically", "Logical/Analytical"),
            ("Group study sessions", "Social/Interpersonal"),
            ("Studying alone with focus", "Solitary/Intrapersonal"),
            ("Relating to practical applications", "Naturalist")
        ]
    },
    {
        "question": "What type of explanation clarifies technical concepts best?",
        "options": [
            ("Graphical models or simulations", "Visual"),
            ("Verbal explanations or Q&A", "Auditory"),
            ("Detailed documentation or code", "Reading/Writing"),
            ("Hands-on coding or experiments", "Kinesthetic"),
            ("Mathematical derivations or logic", "Logical/Analytical"),
            ("Peer discussions or brainstorming", "Social/Interpersonal"),
            ("Self-guided problem-solving", "Solitary/Intrapersonal"),
            ("Real-world system examples", "Naturalist")
        ]
    },
    {
        "question": "When reviewing BTech material, what works best?",
        "options": [
            ("Watching tutorial videos", "Visual"),
            ("Listening to audio summaries", "Auditory"),
            ("Rereading notes or textbooks", "Reading/Writing"),
            ("Practical coding or lab work", "Kinesthetic"),
            ("Analyzing algorithms or proofs", "Logical/Analytical"),
            ("Discussing with classmates", "Social/Interpersonal"),
            ("Reflecting alone on concepts", "Solitary/Intrapersonal"),
            ("Observing industry applications", "Naturalist")
        ]
    },
    {
        "question": "What makes a technical lecture engaging for you?",
        "options": [
            ("Slides with diagrams or animations", "Visual"),
            ("Clear verbal explanations", "Auditory"),
            ("Handouts or detailed notes", "Reading/Writing"),
            ("Interactive labs or demos", "Kinesthetic"),
            ("Logical problem breakdowns", "Logical/Analytical"),
            ("Group discussions or Q&A", "Social/Interpersonal"),
            ("Time for personal reflection", "Solitary/Intrapersonal"),
            ("Real-world tech examples", "Naturalist")
        ]
    },
    {
        "question": "How do you tackle a complex engineering problem?",
        "options": [
            ("Drawing diagrams or flowcharts", "Visual"),
            ("Talking through steps aloud", "Auditory"),
            ("Writing pseudocode or steps", "Reading/Writing"),
            ("Prototyping or testing solutions", "Kinesthetic"),
            ("Using logical reasoning or math", "Logical/Analytical"),
            ("Collaborating with peers", "Social/Interpersonal"),
            ("Thinking through alone", "Solitary/Intrapersonal"),
            ("Relating to system patterns", "Naturalist")
        ]
    },
    {
        "question": "What study environment suits your BTech learning?",
        "options": [
            ("Space with screens or visuals", "Visual"),
            ("Quiet room for audio focus", "Auditory"),
            ("Library with books and notes", "Reading/Writing"),
            ("Lab with tools or computers", "Kinesthetic"),
            ("Organized desk for analysis", "Logical/Analytical"),
            ("Group study room", "Social/Interpersonal"),
            ("Private study corner", "Solitary/Intrapersonal"),
            ("Outdoor or tech-inspired space", "Naturalist")
        ]
    },
    {
        "question": "Which feedback method improves your BTech skills most?",
        "options": [
            ("Visual performance charts", "Visual"),
            ("Verbal feedback from mentors", "Auditory"),
            ("Written code reviews or reports", "Reading/Writing"),
            ("Hands-on project critiques", "Kinesthetic"),
            ("Logical error analysis", "Logical/Analytical"),
            ("Group feedback sessions", "Social/Interpersonal"),
            ("Self-assessment of work", "Solitary/Intrapersonal"),
            ("Real-world application feedback", "Naturalist")
        ]
    },
    {
        "question": "How do you prefer to practice coding or technical skills?",
        "options": [
            ("Using visual debuggers or IDEs", "Visual"),
            ("Listening to coding tutorials", "Auditory"),
            ("Reading code or documentation", "Reading/Writing"),
            ("Writing and testing code", "Kinesthetic"),
            ("Analyzing code logic", "Logical/Analytical"),
            ("Pair programming with peers", "Social/Interpersonal"),
            ("Coding alone in focus", "Solitary/Intrapersonal"),
            ("Applying code to real-world scenarios", "Naturalist")
        ]
    },
    {
        "question": "What motivates you most in a BTech project?",
        "options": [
            ("Visualizing project outcomes", "Visual"),
            ("Discussing ideas with the team", "Auditory"),
            ("Documenting project plans", "Reading/Writing"),
            ("Building prototypes or models", "Kinesthetic"),
            ("Solving technical challenges", "Logical/Analytical"),
            ("Collaborating with teammates", "Social/Interpersonal"),
            ("Working independently on tasks", "Solitary/Intrapersonal"),
            ("Connecting to real-world impact", "Naturalist")
        ]
    },
    {
        "question": "How do you understand complex algorithms best?",
        "options": [
            ("Visualizing with flowcharts", "Visual"),
            ("Hearing explanations or lectures", "Auditory"),
            ("Reading algorithm pseudocode", "Reading/Writing"),
            ("Implementing algorithms in code", "Kinesthetic"),
            ("Deriving or analyzing logic", "Logical/Analytical"),
            ("Discussing with study groups", "Social/Interpersonal"),
            ("Studying algorithms alone", "Solitary/Intrapersonal"),
            ("Relating to practical systems", "Naturalist")
        ]
    },
    {
        "question": "What helps you debug code effectively?",
        "options": [
            ("Using visual debugging tools", "Visual"),
            ("Talking through errors aloud", "Auditory"),
            ("Reading error logs or docs", "Reading/Writing"),
            ("Testing fixes in real-time", "Kinesthetic"),
            ("Logically tracing errors", "Logical/Analytical"),
            ("Asking peers for insights", "Social/Interpersonal"),
            ("Debugging alone quietly", "Solitary/Intrapersonal"),
            ("Comparing to real-world cases", "Naturalist")
        ]
    },
    {
        "question": "How do you prefer to present your BTech project?",
        "options": [
            ("Using slides or visuals", "Visual"),
            ("Explaining verbally", "Auditory"),
            ("Writing detailed reports", "Reading/Writing"),
            ("Demonstrating a working model", "Kinesthetic"),
            ("Breaking down logic clearly", "Logical/Analytical"),
            ("Presenting as a team", "Social/Interpersonal"),
            ("Preparing solo presentations", "Solitary/Intrapersonal"),
            ("Showing real-world relevance", "Naturalist")
        ]
    },
    {
        "question": "What type of BTech assignment do you enjoy most?",
        "options": [
            ("Creating diagrams or simulations", "Visual"),
            ("Participating in discussions", "Auditory"),
            ("Writing code or essays", "Reading/Writing"),
            ("Hands-on experiments or coding", "Kinesthetic"),
            ("Solving logical problems", "Logical/Analytical"),
            ("Group-based projects", "Social/Interpersonal"),
            ("Individual assignments", "Solitary/Intrapersonal"),
            ("Fieldwork or case studies", "Naturalist")
        ]
    },
    {
        "question": "How do you approach learning new software tools?",
        "options": [
            ("Watching tutorial videos", "Visual"),
            ("Listening to guided walkthroughs", "Auditory"),
            ("Reading manuals or guides", "Reading/Writing"),
            ("Experimenting with the tool", "Kinesthetic"),
            ("Analyzing tool logic", "Logical/Analytical"),
            ("Learning with peers", "Social/Interpersonal"),
            ("Exploring alone", "Solitary/Intrapersonal"),
            ("Applying to real tasks", "Naturalist")
        ]
    },
    {
        "question": "What helps you memorize technical terms?",
        "options": [
            ("Using flashcards or visuals", "Visual"),
            ("Repeating terms aloud", "Auditory"),
            ("Writing definitions", "Reading/Writing"),
            ("Using terms in practice", "Kinesthetic"),
            ("Linking terms to logic", "Logical/Analytical"),
            ("Discussing with others", "Social/Interpersonal"),
            ("Reviewing alone quietly", "Solitary/Intrapersonal"),
            ("Connecting to real-world use", "Naturalist")
        ]
    },
    {
        "question": "How do you stay focused during BTech studies?",
        "options": [
            ("Using visual study aids", "Visual"),
            ("Listening to study music", "Auditory"),
            ("Reading or note-taking", "Reading/Writing"),
            ("Working in a hands-on lab", "Kinesthetic"),
            ("Organizing tasks logically", "Logical/Analytical"),
            ("Studying with friends", "Social/Interpersonal"),
            ("Isolating in a quiet space", "Solitary/Intrapersonal"),
            ("Studying in natural settings", "Naturalist")
        ]
    },
    {
        "question": "What inspires you to excel in BTech coursework?",
        "options": [
            ("Visualizing success or goals", "Visual"),
            ("Hearing success stories", "Auditory"),
            ("Reading motivational texts", "Reading/Writing"),
            ("Building tangible projects", "Kinesthetic"),
            ("Solving challenging problems", "Logical/Analytical"),
            ("Team achievements", "Social/Interpersonal"),
            ("Personal goal-setting", "Solitary/Intrapersonal"),
            ("Real-world tech impact", "Naturalist")
        ]
    }
]

tie_breaker_questions = {
    "Visual": [
        {
            "question": "When learning a new programming concept, do you prefer visualizing it?",
            "options": [("Creating a mental or drawn diagram", "Visual"), ("Other methods", "Other")]
        },
        {
            "question": "Do you find visual aids like graphs most effective for understanding data?",
            "options": [("Yes, I rely on visuals", "Visual"), ("No, I use other approaches", "Other")]
        }
    ],
    "Auditory": [
        {
            "question": "Do you learn best by discussing a technical topic aloud?",
            "options": [("Yes, talking helps me understand", "Auditory"), ("No, I prefer other ways", "Other")]
        },
        {
            "question": "Do audio lectures clarify complex BTech concepts for you?",
            "options": [("Yes, I learn best by listening", "Auditory"), ("No, I use other methods", "Other")]
        }
    ],
    "Reading/Writing": [
        {
            "question": "Do you prefer writing notes to understand technical material?",
            "options": [("Yes, writing helps me process", "Reading/Writing"), ("No, I prefer other methods", "Other")]
        },
        {
            "question": "Do you learn best by reading detailed technical documentation?",
            "options": [("Yes, reading is most effective", "Reading/Writing"), ("No, I use other approaches", "Other")]
        }
    ],
    "Kinesthetic": [
        {
            "question": "Do you learn best by physically engaging with tools or code?",
            "options": [("Yes, hands-on practice is key", "Kinesthetic"), ("No, I prefer other methods", "Other")]
        }
    ],
    "Logical/Analytical": [
        {
            "question": "Do you enjoy deriving solutions logically for BTech problems?",
            "options": [("Yes, logical analysis is my strength", "Logical/Analytical"), ("No, I prefer other methods", "Other")]
        }
    ],
    "Social/Interpersonal": [
        {
            "question": "Do you learn best through group discussions or teamwork?",
            "options": [("Yes, collaboration helps me", "Social/Interpersonal"), ("No, I prefer other methods", "Other")]
        }
    ],
    "Solitary/Intrapersonal": [
        {
            "question": "Do you prefer studying technical topics alone?",
            "options": [("Yes, solo study suits me", "Solitary/Intrapersonal"), ("No, I prefer other methods", "Other")]
        }
    ],
    "Naturalist": [
        {
            "question": "Do you learn best by relating concepts to real-world systems?",
            "options": [("Yes, real-world connections help", "Naturalist"), ("No, I prefer other methods", "Other")]
        }
    ]
}

category_to_index = {
    "Visual": 0,
    "Auditory": 1,
    "Reading/Writing": 2,
    "Kinesthetic": 3,
    "Logical/Analytical": 4,
    "Social/Interpersonal": 5,
    "Solitary/Intrapersonal": 6,
    "Naturalist": 7
}
index_to_category = {v: k for k, v in category_to_index.items()}

def ask_questions(num_questions=20):
    answers = []
    raw_answers = []
    for i, q in enumerate(questions[:num_questions]):
        print(f"\nQ{i+1}. {q['question']}")
        for j, (text, _) in enumerate(q['options']):
            print(f"  {j+1}. {text}")

        retries = 0
        while retries < 3:
            try:
                choice = input("Select an option (1-8): ").strip()
                if choice.isdigit():
                    choice = int(choice)
                    if 1 <= choice <= 8:
                        break
                print("Please enter a valid number (1-8). Try again.")
                retries += 1
            except Exception as e:
                print("Error:", e)
                retries += 1
        else:
            print("Too many invalid attempts. Exiting quiz.")
            sys.exit(1)

        raw_answers.append(choice)
        _, category = q['options'][choice - 1]
        answer_vector = [0] * 8
        answer_vector[category_to_index[category]] = 1
        answers.extend(answer_vector)

    return answers, raw_answers

def ask_tie_breaker_questions(tied_styles, max_additional=5):
    additional_answers = []
    style_counts = {style: 0 for style in tied_styles}
    question_count = 0

    print("\nTie detected! Asking additional questions to determine your dominant learning style.")
    for _ in range(max_additional):
        if question_count >= max_additional:
            break
        style = random.choice(tied_styles)
        available_questions = tie_breaker_questions.get(style, [])
        if not available_questions:
            continue
        q = random.choice(available_questions)
        print(f"\nTie-breaker Q{question_count+1}. {q['question']}")
        for j, (text, _) in enumerate(q['options']):
            print(f"  {j+1}. {text}")

        retries = 0
        while retries < 3:
            try:
                choice = input("Select an option (1-2): ").strip()
                if choice.isdigit():
                    choice = int(choice)
                    if 1 <= choice <= 2:
                        break
                print("Please enter a valid number (1-2). Try again.")
                retries += 1
            except Exception as e:
                print("Error:", e)
                retries += 1
        else:
            print("Too many invalid attempts. Skipping this question.")
            continue

        additional_answers.append((choice, q['options']))
        _, selected_style = q['options'][choice - 1]
        if selected_style in style_counts:
            style_counts[selected_style] += 1
        question_count += 1

    max_count = max(style_counts.values(), default=0)
    if max_count == 0:
        winner = min(tied_styles, key=lambda x: list(category_to_index.keys()).index(x))
    else:
        winners = [style for style, count in style_counts.items() if count == max_count]
        winner = min(winners, key=lambda x: list(category_to_index.keys()).index(x))
    
    return winner, additional_answers

def generate_synthetic_data():
    synthetic_data = []
    for style_idx in range(8):
        for _ in range(200):
            user_data = []
            secondary_idx = random.choice([i for i in range(8) if i != style_idx])
            for _ in range(20):
                answer_vector = [0] * 8
                rand = random.random()
                if rand < 0.98:
                    answer_vector[style_idx] = 1
                elif rand < 0.995:
                    answer_vector[secondary_idx] = 1
                else:
                    answer_vector[random.randint(0, 7)] = 1
                user_data.extend(answer_vector)
            synthetic_data.append(user_data)
    return np.array(synthetic_data)

def build_model():
    data = generate_synthetic_data()
    kmeans = KMeans(n_clusters=8, random_state=42, n_init=30, max_iter=1000)
    kmeans.fit(data)
    score = silhouette_score(data, kmeans.labels_)
    print(f"Clustering quality (silhouette score): {score:.3f}")
    return kmeans, data

def api_classify_learning_style(questions, raw_answers, additional_answers=None, tied_styles=None):
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        prompt = """
        You are an expert in learning style classification for BTech students. Given 20 questions about learning preferences, each with 8 options mapped to Visual, Auditory, Reading/Writing, Kinesthetic, Logical/Analytical, Social/Interpersonal, Solitary/Intrapersonal, and Naturalist, and the user's selected options (1-8), count the occurrences of each style and compute percentages (e.g., 6 Visual selections = 30%). If additional tie-breaker questions are provided, incorporate their counts to resolve ties. Return a JSON object with 'percentages' (e.g., {'Visual': 30, 'Auditory': 30, ...}) and 'description' (e.g., 'Visual-cum-Auditory'). For the description, select the style with the highest count as primary and the second-highest as secondary; in case of ties, use the order: Visual, Auditory, Reading/Writing, Kinesthetic, Logical/Analytical, Social/Interpersonal, Solitary/Intrapersonal, Naturalist. Ensure percentages sum to 100.
        """
        for i, q in enumerate(questions):
            prompt += f"Question {i+1}: {q['question']}\n"
            for j, (text, style) in enumerate(q['options']):
                prompt += f"  {j+1}. {text} ({style})\n"
            prompt += f"User selected option: {raw_answers[i]}\n"
        
        if additional_answers and tied_styles:
            prompt += "\nAdditional tie-breaker questions for styles: " + ", ".join(tied_styles) + "\n"
            for i, (choice, options) in enumerate(additional_answers):
                prompt += f"Tie-breaker Question {i+1}: {options[0][1]} vs Other\n"
                prompt += f"  1. {options[0][0]} ({options[0][1]})\n  2. {options[1][0]} ({options[1][1]})\n"
                prompt += f"User selected option: {choice}\n"
        
        prompt += "\nReturn a JSON object with 'percentages' (style percentages summing to 100) and 'description' (Primary-cum-Secondary)."
        response = model.generate_content(prompt)
        result = response.text.strip()
        result = re.sub(r'^```json\n|```$|^```.*\n|```', '', result, flags=re.MULTILINE)
        result = result.strip()
        result = json.loads(result)
        percentages = result.get('percentages', {})
        description = result.get('description', '')
        if not all(k in category_to_index for k in percentages) or len(percentages) != 8:
            raise ValueError(f"Invalid percentages: {percentages}")
        if abs(sum(percentages.values()) - 100) > 1:
            raise ValueError(f"Percentages do not sum to ~100: {percentages}")
        if not description or '-' not in description:
            raise ValueError(f"Invalid description: {description}")
        return percentages, description
    except Exception as e:
        print(f"API classification failed: {e}. Falling back to rule-based prediction.")
        return None, None

def rule_based_predict(raw_answers, additional_answers=None, tied_styles=None):
    style_counts = [0] * 8
    for i, choice in enumerate(raw_answers):
        _, category = questions[i]['options'][int(choice) - 1]
        style_counts[category_to_index[category]] += 1
    
    if additional_answers and tied_styles:
        for choice, options in additional_answers:
            _, category = options[choice - 1]
            if category in tied_styles:
                style_counts[category_to_index[category]] += 1
    
    total_answers = len(raw_answers) + (len(additional_answers) if additional_answers else 0)
    percentages = {index_to_category[i]: (count / total_answers * 100) for i, count in enumerate(style_counts)}
    sorted_styles = sorted(percentages.items(), key=lambda x: (-x[1], list(category_to_index.keys()).index(x[0])))
    description = f"{sorted_styles[0][0]}-cum-{sorted_styles[1][0]}"
    return percentages, description, style_counts

def kmeans_predict(user_vector, kmeans, data):
    user_vector = np.array(user_vector).reshape(1, -1)
    cluster = kmeans.predict(user_vector)[0]
    cluster_data = data[kmeans.labels_ == cluster]
    mean_vector = np.mean(cluster_data, axis=0)
    dominant_style = np.argmax(np.sum(mean_vector.reshape(20, 8), axis=0))
    return index_to_category[dominant_style]

def plot_stacked_bar_chart(raw_answers, additional_answers=None, tied_styles=None):
    style_counts_per_question = np.zeros((20 + (len(additional_answers) if additional_answers else 0), 8))
    for i, choice in enumerate(raw_answers):
        _, category = questions[i]['options'][int(choice) - 1]
        style_counts_per_question[i, category_to_index[category]] = 1
    if additional_answers and tied_styles:
        for j, (choice, options) in enumerate(additional_answers):
            _, category = options[choice - 1]
            if category in tied_styles:
                style_counts_per_question[20 + j, category_to_index[category]] = 1
    
    labels = [f"Q{i+1}" for i in range(20)] + ([f"T{i+1}" for i in range(len(additional_answers))] if additional_answers else [])
    styles = list(category_to_index.keys())
    counts = [style_counts_per_question[:, i].tolist() for i in range(8)]
    
    pltxt.clear_figure()
    pltxt.stacked_bar(labels, counts)
    pltxt.title("Learning Style Contributions by Question")
    pltxt.xlabel("Questions")
    pltxt.ylabel("Selections")
    pltxt.plot_size(width=60, height=15)
    pltxt.show()
    print("Legend: " + ", ".join(f"{i+1}={styles[i]}" for i in range(8)))

def plot_radar_chart(percentages):
    labels = ['Vis', 'Aud', 'R/W', 'Kin', 'Log', 'Soc', 'Sol', 'Nat']
    values = [percentages.get(index_to_category[i], 0) for i in range(8)]
    values += [values[0]]  # Close the radar loop
    angles = np.linspace(0, 2 * np.pi, len(labels) + 1, endpoint=True)
    
    pltxt.clear_figure()
    pltxt.plot(angles, values, style='solid', marker='dot')
    pltxt.scatter(angles[:-1], values[:-1], marker='dot')
    for i, label in enumerate(labels):
        pltxt.text(label, x=angles[i], y=values[i] + 5, alignment='center')
    pltxt.title("Learning Style Radar Chart")
    pltxt.plot_size(width=50, height=15)
    pltxt.show()

def plot_terminal_cluster_chart(user_vector, data, kmeans):
    pca = PCA(n_components=2, random_state=42)
    data_reduced = pca.fit_transform(data)
    labels = kmeans.labels_
    user_point = pca.transform([user_vector])[0]

    cluster_styles = []
    for i in range(8):
        cluster_data = data[labels == i]
        if len(cluster_data) > 0:
            mean_vector = np.mean(cluster_data, axis=0)
            dominant_style = np.argmax(np.sum(mean_vector.reshape(20, 8), axis=0))
            cluster_styles.append(index_to_category[dominant_style])
        else:
            cluster_styles.append(index_to_category[i])

    pltxt.clear_figure()
    for i in range(8):
        cluster_points = data_reduced[labels == i]
        for x, y in cluster_points:
            pltxt.scatter([x], [y], marker='o', color=i + 1)
    pltxt.scatter([user_point[0]], [user_point[1]], marker='*', color='white')
    pltxt.title("Cluster Scatter Plot (PCA Projection)")
    pltxt.xlabel("PCA Component 1")
    pltxt.ylabel("PCA Component 2")
    pltxt.plot_size(width=60, height=15)
    pltxt.show()
    print("Legend: * = Your Input, o = Cluster Points (Colors: 1=Visual, 2=Auditory, 3=R/W, 4=Kin, 5=Log, 6=Soc, 7=Sol, 8=Nat)")

def plot_pie_chart(percentages):
    labels = ['Vis', 'Aud', 'R/W', 'Kin', 'Log', 'Soc', 'Sol', 'Nat']
    values = [percentages.get(index_to_category[i], 0) for i in range(8)]
    
    pltxt.clear_figure()
    pltxt.bar(labels, values, orientation='horizontal', width=0.8)
    pltxt.title("Learning Style Percentage Distribution")
    pltxt.xlabel("Percentage (%)")
    pltxt.ylabel("Styles")
    pltxt.plot_size(width=50, height=15)
    pltxt.show()

def plot_bar_with_error_bars(percentages, style_counts, total_answers):
    labels = ['Vis', 'Aud', 'R/W', 'Kin', 'Log', 'Soc', 'Sol', 'Nat']
    values = [percentages.get(index_to_category[i], 0) for i in range(8)]
    errors = [np.sqrt((count / total_answers) * (1 - count / total_answers) / total_answers) * 100 if count > 0 else 0 for count in style_counts]
    
    pltxt.clear_figure()
    pltxt.bar(labels, values, orientation='vertical', width=0.4)
    for i, (value, error) in enumerate(zip(values, errors)):
        if error > 0:
            pltxt.plot([i + 1, i + 1], [value - error, value + error], color='white', marker='|')
    pltxt.title("Learning Style Percentages with Confidence Intervals")
    pltxt.xlabel("Styles")
    pltxt.ylabel("Percentage (%)")
    pltxt.plot_size(width=50, height=15)
    pltxt.show()

def main():
    print("Welcome to the BTech Learning Style Quiz\n")
    model, data = build_model()
    user_vector, raw_answers = ask_questions(num_questions=20)
    
    style_counts = [0] * 8
    for i, choice in enumerate(raw_answers):
        _, category = questions[i]['options'][int(choice) - 1]
        style_counts[category_to_index[category]] += 1
    max_count = max(style_counts)
    tied_styles = [index_to_category[i] for i, count in enumerate(style_counts) if count == max_count and count > 0]
    
    additional_answers = None
    winner = None
    if len(tied_styles) > 1:
        winner, additional_answers = ask_tie_breaker_questions(tied_styles)
        print(f"\nTie resolved! Dominant style: {winner}")
    
    api_percentages, api_description = api_classify_learning_style(questions, raw_answers, additional_answers, tied_styles)
    rule_percentages, rule_description, style_counts = rule_based_predict(raw_answers, additional_answers, tied_styles)
    kmeans_style = kmeans_predict(user_vector, model, data)
    
    print("\nYour learning style gradient (API-based):")
    if api_percentages:
        for style, percent in api_percentages.items():
            print(f"{style}: {percent:.1f}%")
        print(f"Description: {api_description}")
    else:
        print("N/A (API failed)")
    
    print("\nYour learning style gradient (Rule-based):")
    for style, percent in rule_percentages.items():
        print(f"{style}: {percent:.1f}%")
    print(f"Description: {rule_description}")
    
    print("\nYour learning style (KMeans-based):")
    print(f"Dominant style: {kmeans_style}")
    
    if api_percentages and api_description != rule_description:
        print("\nNote: API and rule-based descriptions differ. Rule-based is based on highest selection count.")
    
    print("\nAnswer distribution:")
    total_answers = 20 + (len(additional_answers) if additional_answers else 0)
    for i, count in enumerate(style_counts):
        print(f"{index_to_category[i]}: {count}/{total_answers} selections")
    
    print("\nStacked Bar Chart (Question Contributions):")
    plot_stacked_bar_chart(raw_answers, additional_answers, tied_styles)
    print("\nRadar Chart (Style Percentages):")
    plot_radar_chart(rule_percentages if api_percentages is None else api_percentages)
    print("\nCluster Scatter Plot:")
    plot_terminal_cluster_chart(user_vector, data, model)
    print("\nPie Chart (Percentage Distribution):")
    plot_pie_chart(rule_percentages if api_percentages is None else api_percentages)
    print("\nBar Chart with Confidence Intervals:")
    plot_bar_with_error_bars(rule_percentages if api_percentages is None else api_percentages, style_counts, total_answers)

if __name__ == "__main__":
    main()

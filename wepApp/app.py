from flask import Flask, render_template, request, redirect, url_for, session
from Database.db import get_db_connection
from werkzeug.security import generate_password_hash, check_password_hash
import pandas as pd
import os

app = Flask(__name__)

app.secret_key = "fairvote_secret_key"


# =========================================================
# HOME
# =========================================================

@app.route("/")
def home():
    return render_template("index.html")


# =========================================================
# REGISTER
# =========================================================

@app.route("/register", methods=["GET", "POST"])
def register():

    if request.method == "POST":

        username = request.form["username"]
        password = request.form["password"]

        password_hash = generate_password_hash(password)

        connection = get_db_connection()
        cursor = connection.cursor()

        try:

            cursor.execute(
                """
                INSERT INTO users (username, password_hash, role)
                VALUES (%s, %s, %s)
                """,
                (username, password_hash, "voter")
            )

            connection.commit()

        except Exception as e:

            connection.rollback()

            cursor.close()
            connection.close()

            return f"Registration error: {e}"

        cursor.close()
        connection.close()

        return redirect(url_for("login"))

    return render_template("register.html")


# =========================================================
# LOGIN
# =========================================================

@app.route("/login", methods=["GET", "POST"])
def login():

    if request.method == "POST":

        username = request.form["username"]
        password = request.form["password"]

        connection = get_db_connection()
        cursor = connection.cursor(dictionary=True)

        cursor.execute(
            "SELECT * FROM users WHERE username = %s",
            (username,)
        )

        user = cursor.fetchone()

        cursor.close()
        connection.close()

        # Check username and password
        if user and check_password_hash(
            user["password_hash"],
            password
        ):

            # Store user information
            session["user_id"] = user["user_id"]
            session["username"] = user["username"]
            session["role"] = user["role"]

            # =================================================
            # IMPORTANT:
            # Redirect admin and voter to different dashboards
            # =================================================

            if user["role"] == "admin":

                return redirect(url_for("admin_dashboard"))

            else:

                return redirect(url_for("dashboard"))

        return render_template(
            "login.html",
            error="Invalid username or password"
        )

    return render_template("login.html")


# =========================================================
# VOTER DASHBOARD
# =========================================================

@app.route("/dashboard")
def dashboard():

    # Must be logged in
    if "user_id" not in session:
        return redirect(url_for("login"))

    # Admin should not access voter dashboard
    if session.get("role") == "admin":
        return redirect(url_for("admin_dashboard"))

    connection = get_db_connection()
    cursor = connection.cursor(dictionary=True)

    # Get active election
    cursor.execute("""
        SELECT *
        FROM elections
        WHERE status = 'active'
        LIMIT 1
    """)

    election = cursor.fetchone()

    candidates = []

    if election:

        cursor.execute("""
            SELECT
                candidate_id,
                candidate_name,
                party_name
            FROM candidates
            WHERE election_id = %s
        """, (election["election_id"],))

        candidates = cursor.fetchall()

    cursor.close()
    connection.close()

    return render_template(
        "dashboard.html",
        election=election,
        candidates=candidates
    )


# =========================================================
# VOTING
# =========================================================

@app.route("/vote", methods=["GET", "POST"])
def vote():

    # Check login first
    if "user_id" not in session:
        return redirect(url_for("login"))

    # Admin cannot vote
    if session.get("role") == "admin":
        return redirect(url_for("admin_dashboard"))

    connection = get_db_connection()
    cursor = connection.cursor(dictionary=True)

    # =====================================================
    # POST → User submits vote
    # =====================================================

    if request.method == "POST":

        candidate_id = request.form.get("candidate_id")

        voter_id = session["user_id"]

        # Get active election
        cursor.execute("""
            SELECT *
            FROM elections
            WHERE status = 'active'
            LIMIT 1
        """)

        election = cursor.fetchone()

        if not election:

            cursor.close()
            connection.close()

            return "No active election available."

        election_id = election["election_id"]

        # =================================================
        # Check duplicate vote
        # =================================================

        cursor.execute("""
            SELECT vote_id
            FROM votes
            WHERE election_id = %s
            AND voter_id = %s
        """, (election_id, voter_id))

        existing_vote = cursor.fetchone()

        if existing_vote:

            cursor.close()
            connection.close()

            return "You have already voted in this election."

        # =================================================
        # Verify candidate
        # =================================================

        cursor.execute("""
            SELECT candidate_id
            FROM candidates
            WHERE candidate_id = %s
            AND election_id = %s
        """, (candidate_id, election_id))

        candidate = cursor.fetchone()

        if not candidate:

            cursor.close()
            connection.close()

            return "Invalid candidate selection."

        # =================================================
        # Save vote
        # =================================================

        cursor.execute("""
            INSERT INTO votes
            (election_id, voter_id, candidate_id)
            VALUES (%s, %s, %s)
        """, (election_id, voter_id, candidate_id))

        connection.commit()

        cursor.close()
        connection.close()

        return render_template("vote_success.html")

    # =====================================================
    # GET → Display voting page
    # =====================================================

    cursor.execute("""
        SELECT *
        FROM elections
        WHERE status = 'active'
        LIMIT 1
    """)

    election = cursor.fetchone()

    if not election:

        cursor.close()
        connection.close()

        return "No active election available."

    # Get candidates
    cursor.execute("""
        SELECT *
        FROM candidates
        WHERE election_id = %s
    """, (election["election_id"],))

    candidates = cursor.fetchall()

    cursor.close()
    connection.close()

    return render_template(
        "vote.html",
        election=election,
        candidates=candidates
    )


# =========================================================
# ADMIN DASHBOARD
# =========================================================

@app.route("/admin/dashboard")
def admin_dashboard():

    # ==========================================
    # 1. Check login
    # ==========================================

    if "user_id" not in session:
        return redirect(url_for("login"))


    # ==========================================
    # 2. Check admin role
    # ==========================================

    if session.get("role") != "admin":
        return "Access denied", 403


    # ==========================================
    # 3. Database connection
    # ==========================================

    connection = get_db_connection()
    cursor = connection.cursor(dictionary=True)


    # ==========================================
    # 4. Get active election
    # ==========================================

    cursor.execute("""
        SELECT *
        FROM elections
        WHERE status = 'active'
        LIMIT 1
    """)

    election = cursor.fetchone()


    # ==========================================
    # 5. Total voters
    # ==========================================

    cursor.execute("""
        SELECT COUNT(*) AS total
        FROM users
        WHERE role = 'voter'
    """)

    total_voters = cursor.fetchone()["total"]


    # ==========================================
    # 6. Total votes
    # ==========================================

    cursor.execute("""
        SELECT COUNT(*) AS total
        FROM votes
    """)

    total_votes = cursor.fetchone()["total"]


    # ==========================================
    # 7. Calculate turnout
    # ==========================================

    if total_voters > 0:

        turnout = round(
            (total_votes / total_voters) * 100,
            2
        )

    else:

        turnout = 0


    # ==========================================
    # 8. Candidate results
    # ==========================================

    results = []

    if election:

        cursor.execute("""
            SELECT
                c.candidate_name,
                c.party_name,
                COUNT(v.vote_id) AS vote_count

            FROM candidates c

            LEFT JOIN votes v
                ON c.candidate_id = v.candidate_id

            WHERE c.election_id = %s

            GROUP BY
                c.candidate_id,
                c.candidate_name,
                c.party_name

            ORDER BY vote_count DESC
        """, (election["election_id"],))

        results = cursor.fetchall()


    # ==========================================
    # 9. Close database
    # ==========================================

    cursor.close()
    connection.close()


    # ==========================================
    # 10. Load ML anomaly results
    # ==========================================

    anomalies = []
    anomaly_count = 0


    # Path of anomaly results file

    anomaly_file = os.path.join(
        "data",
        "anomaly_results.csv"
    )


    if os.path.exists(anomaly_file):

        try:

            anomaly_data = pd.read_csv(
                anomaly_file
            )


            # Select only detected anomalies

            anomaly_data = anomaly_data[
                anomaly_data["ml_anomaly"] == 1
            ]


            # Number of anomalies

            anomaly_count = len(
                anomaly_data
            )


            # Convert to dictionary
            # so Jinja can display it

            anomalies = anomaly_data[
                [
                    "booth_id",
                    "turnout_percentage",
                    "candidate_A_share",
                    "candidate_B_share",
                    "candidate_C_share",
                    "candidate_D_share",
                    "rejected_vote_rate",
                    "anomaly_score"
                ]
            ].to_dict(
                orient="records"
            )


        except Exception as e:

            print(
                "Error loading anomaly results:",
                e
            )



    # ==========================================
    # 11. Send everything to dashboard
    # ==========================================

    return render_template(
        "admin_dashboard.html",

        total_voters=total_voters,

        total_votes=total_votes,

        turnout=turnout,

        election=election,

        results=results,

        anomalies=anomalies,

        anomaly_count=anomaly_count
    )


# =========================================================
# LOGOUT
# =========================================================

@app.route("/logout")
def logout():

    session.clear()

    return redirect(url_for("login"))


# =========================================================
# RUN APP
# =========================================================

if __name__ == "__main__":
    app.run(debug=True)
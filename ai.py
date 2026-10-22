import streamlit as st
import mysql.connector
from mysql.connector import Error
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from datetime import date
import re

st.set_page_config(
    page_title="Lost & Found AI",
    page_icon="🔎",
    layout="wide"
)

st.title("🔎 Lost & Found AI")
st.caption("A Class 12 AI project that helps people find possible matches for lost belongings.")

# Read MySQL settings from .streamlit/secrets.toml
def get_connection():
    """Connect to the MySQL database using settings from secrets.toml."""
    settings = st.secrets["mysql"]
    return mysql.connector.connect(
        host=settings["host"],
        port=int(settings.get("port", 3306)),
        user=settings["user"],
        password=settings["password"],
        database=settings["database"]
    )

def load_reports():
    """Load saved reports from MySQL."""
    connection = get_connection()
    cursor = None

    try:
        query = """
            SELECT id, report_type, item_name, category, description,
                   colour, location, report_date, status, created_at
            FROM reports
            ORDER BY created_at DESC
        """

        cursor = connection.cursor()
        cursor.execute(query)

        rows = cursor.fetchall()
        columns = [column[0] for column in cursor.description]

        return pd.DataFrame(rows, columns=columns)

    finally:
        if cursor is not None:
            cursor.close()
        connection.close()

def save_report(report):
    """Save one lost or found report to MySQL."""
    connection = get_connection()
    try:
        cursor = connection.cursor()
        query = """
            INSERT INTO reports
            (report_type, item_name, category, description, colour,
             location, report_date, contact_name, contact_email)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        values = (
            report["report_type"],
            report["item_name"],
            report["category"],
            report["description"],
            report["colour"],
            report["location"],
            report["report_date"],
            report["contact_name"],
            report["contact_email"]
        )
        cursor.execute(query, values)
        connection.commit()
    finally:
        connection.close()

def mark_reports_returned(lost_id, found_id):
    """Close a verified lost/found pair together after the item is handed back."""
    lost_id, found_id = int(lost_id), int(found_id)
    if lost_id == found_id:
        raise ValueError("Choose two different reports.")

    connection = get_connection()
    cursor = None
    try:
        connection.start_transaction()
        cursor = connection.cursor()
        # Lock both records so another return cannot change them halfway through.
        cursor.execute(
            "SELECT id, report_type, status FROM reports "
            "WHERE id IN (%s, %s) FOR UPDATE",
            (lost_id, found_id)
        )
        selected = {
            int(row[0]): (row[1], row[2]) for row in cursor.fetchall()
        }
        if (selected.get(lost_id) != ("LOST", "OPEN") or
                selected.get(found_id) != ("FOUND", "OPEN")):
            raise ValueError(
                "Both reports must still be open: one LOST and one FOUND. "
                "Refresh Browse Reports and check your selection."
            )

        cursor.execute(
            "UPDATE reports SET status = %s "
            "WHERE id IN (%s, %s) AND status = %s",
            ("RETURNED", lost_id, found_id, "OPEN")
        )
        if cursor.rowcount != 2:
            raise ValueError("Both reports could not be updated together.")
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        if cursor is not None:
            cursor.close()
        connection.close()

def prepare_text(text):
    """Make selected words consistent before TF-IDF compares descriptions."""
    text = re.sub(r"[^a-z0-9\s]", " ", text.lower())
    text = " ".join(text.split())

    replacements = {
        # Colours and materials
        "stainless steel": "metal",
        "steel": "metal",
        "navy blue": "blue",
        "navy": "blue",
        "grey": "gray",
        "aluminum": "aluminium",

        # Phones
        "mobile phones": "phone",
        "mobile phone": "phone",
        "cell phones": "phone",
        "cell phone": "phone",
        "cellphones": "phone",
        "cellphone": "phone",
        "smartphones": "phone",
        "smartphone": "phone",
        "phones": "phone",

        # USB drives
        "usb flash drives": "usb drive",
        "usb flash drive": "usb drive",
        "pen drives": "usb drive",
        "pen drive": "usb drive",
        "pendrives": "usb drive",
        "pendrive": "usb drive",
        "thumb drives": "usb drive",
        "thumb drive": "usb drive",
        "flash drives": "usb drive",
        "flash drive": "usb drive",
        "usb drives": "usb drive",

        # Backpacks
        "school bags": "backpack",
        "school bag": "backpack",
        "schoolbags": "backpack",
        "schoolbag": "backpack",
        "back packs": "backpack",
        "back pack": "backpack",
        "rucksacks": "backpack",
        "rucksack": "backpack",
        "knapsacks": "backpack",
        "knapsack": "backpack",
        "backpacks": "backpack",

        # Spectacles
        "prescription glasses": "spectacles",
        "reading glasses": "spectacles",
        "eye glasses": "spectacles",
        "eyeglasses": "spectacles",
        "specs": "spectacles",

        # Other common wording
        "wrist watches": "watch",
        "wrist watch": "watch",
        "wristwatches": "watch",
        "wristwatch": "watch",
        "watches": "watch",
        "note books": "notebook",
        "note book": "notebook",
        "notebooks": "notebook",
        "key chains": "keychain",
        "key chain": "keychain",
        "keychains": "keychain",
        "bottles": "bottle",
        "wallets": "wallet",
        "umbrellas": "umbrella",
        "jewelry": "jewellery",
        "laboratory": "lab",
        "bio": "biology"
    }

    for old, new in replacements.items():
        text = re.sub(r"\b" + re.escape(old) + r"\b", new, text)

    return text

def find_matches(selected_report, all_reports):
    """Use TF-IDF and cosine similarity to rank possible opposite-type reports."""
    if all_reports.empty or selected_report.get("status", "OPEN") != "OPEN":
        return pd.DataFrame()

    opposite_type = "FOUND" if selected_report["report_type"] == "LOST" else "LOST"
    candidates = all_reports[
        (all_reports["report_type"] == opposite_type) &
        (all_reports["status"] == "OPEN")
    ].copy()

    if candidates.empty:
        return candidates

    # Combine several item details into one text description for comparison.
    selected_text = " ".join([
        str(selected_report["item_name"]),
        str(selected_report["category"]),
        str(selected_report["description"]),
        str(selected_report["colour"])
    ])

    selected_text = prepare_text(selected_text)

    candidate_texts = []
    for _, row in candidates.iterrows():
        candidate_text = " ".join([
            str(row["item_name"]),
            str(row["category"]),
            str(row["description"]),
            str(row["colour"])
        ])
        candidate_texts.append(prepare_text(candidate_text))

    all_texts = [selected_text] + candidate_texts

    # TF-IDF converts text into numerical features.
    vectorizer = TfidfVectorizer(stop_words="english")
    try:
        vectors = vectorizer.fit_transform(all_texts)
        scores = cosine_similarity(vectors[0:1], vectors[1:]).flatten()
    except ValueError:
        # This can happen when the descriptions contain no useful words.
        candidates["similarity_score"] = 0.0
        return candidates.head(3)

    candidates["similarity_score"] = scores

    # Give small extra boosts when the category, colour, or location also agrees.
    def extra_score(row):
        bonus = 0.0
        if str(row["category"]).lower() == str(selected_report["category"]).lower():
            bonus += 0.10
        if str(row["colour"]).strip().lower() == str(selected_report["colour"]).strip().lower():
            if str(selected_report["colour"]).strip():
                bonus += 0.05
        selected_location = prepare_text(str(selected_report["location"]))
        candidate_location = prepare_text(str(row["location"]))

        if selected_location and candidate_location == selected_location:
            bonus += 0.05

        return bonus
    candidates["similarity_score"] = candidates.apply(
        lambda row: min(1.0, row["similarity_score"] + extra_score(row)),
        axis=1
    )

    candidates = candidates.sort_values("similarity_score", ascending=False)
    return candidates.head(3)

# Navigation
page = st.sidebar.radio(
    "Navigation",
    ["Home", "Report an Item", "Browse Reports", "Find AI Matches"]
)

if page == "Home":
    st.subheader("Welcome")
    st.write(
        "Report a lost item or something you found. The AI matching section "
        "compares descriptions and ranks up to three possible matches."
    )
    col1, col2, col3 = st.columns(3)
    try:
        reports = load_reports()
        col1.metric("Total reports", len(reports))
        col2.metric("Lost reports", int((reports["report_type"] == "LOST").sum()))
        col3.metric("Found reports", int((reports["report_type"] == "FOUND").sum()))
    except Exception as error:
        st.error("Could not connect to MySQL. Check your setup instructions and database settings.")
        st.caption(f"Technical detail: {error}")

    st.info(
        "Privacy note: use fictional sample contact details while testing. "
        "Do not publish people's private phone numbers or email addresses."
    )

elif page == "Report an Item":
    st.subheader("Submit a Lost or Found Report")
    # Reset widget values before creating the form, and only after a saved report.
    # Invalid entries or database errors leave the user's text in place.
    if st.session_state.pop("_report_saved", False):
        defaults = {
            "report_type_input": "LOST",
            "item_name_input": "",
            "category_input": "Electronics",
            "description_input": "",
            "colour_input": "",
            "location_input": "",
            "report_date_input": date.today(),
            "contact_name_input": "",
            "contact_email_input": ""
        }
        for field, value in defaults.items():
            st.session_state[field] = value
        st.success("Report saved successfully. You can enter another report.")

    if "report_date_input" not in st.session_state:
        st.session_state["report_date_input"] = date.today()

    with st.form("report_form"):
        report_type = st.radio(
            "Report type", ["LOST", "FOUND"], horizontal=True,
            key="report_type_input"
        )
        item_name = st.text_input(
            "Item name", placeholder="e.g. Black wallet", key="item_name_input"
        )
        category = st.selectbox(
            "Category",
            ["Electronics", "Bags", "Wallets", "Keys", "Clothing", "Books",
             "Water bottles", "Jewellery", "Documents", "Other"],
            key="category_input"
        )
        description = st.text_area(
            "Description",
            placeholder="Describe identifying features, marks, or contents. Do not include sensitive details.",
            key="description_input"
        )
        colour = st.text_input("Colour", placeholder="e.g. Black", key="colour_input")
        location = st.text_input(
            "Location", placeholder="e.g. School library", key="location_input"
        )
        report_date = st.date_input(
            "Date lost/found", value=None, key="report_date_input"
        )
        st.markdown("#### Contact details")
        contact_name = st.text_input("Contact name", key="contact_name_input")
        contact_email = st.text_input("Contact email", key="contact_email_input")
        submitted = st.form_submit_button("Save Report", type="primary")

    if submitted:
        if not all([
            item_name.strip(), description.strip(), location.strip(),
            report_date, contact_name.strip(), contact_email.strip()
        ]):
            st.warning("Please complete the item name, description, location, date, and contact fields.")
        elif "@" not in contact_email or "." not in contact_email.split("@")[-1]:
            st.warning("Please enter a valid email address.")
        else:
            new_report = {
                "report_type": report_type,
                "item_name": item_name.strip(),
                "category": category,
                "description": description.strip(),
                "colour": colour.strip(),
                "location": location.strip(),
                "report_date": report_date,
                "contact_name": contact_name.strip(),
                "contact_email": contact_email.strip()
            }
            try:
                save_report(new_report)
            except Exception as error:
                st.error("The report could not be saved. Check the database setup.")
                st.caption(f"Technical detail: {error}")
            else:
                st.session_state["_report_saved"] = True
                st.rerun()

elif page == "Browse Reports":
    st.subheader("Browse Saved Reports")
    return_message = st.session_state.pop("_return_saved", None)
    if return_message:
        st.success(return_message)
    try:
        reports = load_reports()
        if reports.empty:
            st.info("No reports yet. Add a report first.")
        else:
            open_reports = reports[reports["status"] == "OPEN"].copy()
            filter_type = st.selectbox("Filter by report type", ["ALL", "LOST", "FOUND"])
            if filter_type != "ALL":
                reports = reports[reports["report_type"] == filter_type]
            st.dataframe(
                reports[["id", "report_type", "item_name", "category", "colour",
                         "location", "report_date", "status"]],
                hide_index=True
            )
            with st.expander("Record an item return"):
                st.write(
                    "After checking ownership and handing the item back, "
                    "choose the lost report and the found report for that item."
                )
                lost_reports = open_reports[open_reports["report_type"] == "LOST"]
                found_reports = open_reports[open_reports["report_type"] == "FOUND"]
                if lost_reports.empty or found_reports.empty:
                    st.info("Recording a return needs one open LOST report and one open FOUND report.")
                else:
                    lost_choices = {
                        f'#{row["id"]} — {row["item_name"]}': int(row["id"])
                        for _, row in lost_reports.iterrows()
                    }
                    found_choices = {
                        f'#{row["id"]} — {row["item_name"]}': int(row["id"])
                        for _, row in found_reports.iterrows()
                    }
                    lost_label = st.selectbox("Lost report", list(lost_choices))
                    found_label = st.selectbox("Found report", list(found_choices))
                    lost_id = lost_choices[lost_label]
                    found_id = found_choices[found_label]
                    confirmed = st.checkbox(
                        "I have verified ownership and handed the item back.",
                        key=f"return_confirmed_{lost_id}_{found_id}"
                    )
                    if st.button("Mark both reports as returned", disabled=not confirmed):
                        try:
                            mark_reports_returned(lost_id, found_id)
                        except Exception as error:
                            st.error("The return could not be recorded. Please try again.")
                            st.caption(f"Technical detail: {error}")
                        else:
                            st.session_state["_return_saved"] = (
                                f"Reports #{lost_id} and #{found_id} are marked RETURNED. "
                                "They remain in your records and will no longer appear in matching."
                            )
                            st.rerun()
                    st.caption("Record a return only after the item is back with its verified owner.")
    except Exception as error:
        st.error("Could not load reports. Check your MySQL connection.")
        st.caption(f"Technical detail: {error}")

elif page == "Find AI Matches":
    st.subheader("Find Possible Matches")
    try:
        reports = load_reports()
        if reports.empty:
            st.info("Add some lost and found reports before searching.")
        else:
            open_reports = reports[reports["status"] == "OPEN"]
            if open_reports.empty:
                st.info("There are no open reports to match.")
            else:
                labels = {}
                for _, row in open_reports.iterrows():
                    label = f'#{row["id"]} — {row["report_type"]}: {row["item_name"]}'
                    labels[label] = row["id"]

                chosen_label = st.selectbox("Choose a report to find matches for", list(labels.keys()))
                selected_id = labels[chosen_label]
                selected_rows = open_reports[open_reports["id"] == selected_id]
                selected_report = selected_rows.iloc[0]

                st.write("**Description:**", selected_report["description"])
                if st.button("Find up to 3 matches", type="primary"):
                    matches = find_matches(selected_report, reports)
                    if matches.empty:
                        st.warning("No possible matches were found yet.")
                    else:
                        matches = matches[matches["similarity_score"] > 0]
                        if matches.empty:
                            st.warning("No strong text similarities were found. Try adding more descriptive details.")
                        else:
                            st.caption(
                                "Scores are approximate text-similarity estimates, not proof of ownership."
                            )
                            for _, match in matches.iterrows():
                                score = int(round(float(match["similarity_score"]) * 100))
                                with st.container(border=True):
                                    st.markdown(
                                        f'**{match["item_name"]}** — {match["report_type"]}'
                                    )
                                    st.write(match["description"])
                                    st.write(
                                        f'Category: {match["category"]} · '
                                        f'Colour: {match["colour"] or "Not given"}'
                                    )
                                    st.write(
                                        f'Location: {match["location"]} · '
                                        f'Date: {match["report_date"]}'
                                    )
                                    st.progress(min(score, 100), text=f"Similarity score: {score}%")
                                    st.caption(
                                        "This score is a ranking aid. Confirm ownership privately before returning an item."
                                    )
    except Exception as error:
        st.error("Could not run matching. Check your database setup.")
        st.caption(f"Technical detail: {error}")

st.divider()
st.caption("Lost & Found AI • Educational prototype • AI suggestions must be verified by people.")

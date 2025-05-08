# Improved and Modern UI for Swahili Medical Assistant
# Required: pip install SpeechRecognition pyaudio gTTS playsound

import tkinter as tk
import speech_recognition as sr
from gtts import gTTS
import os
from playsound import playsound
import threading

# ==== AI LOGIC ====

def listen_in_swahili():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        chatbox.insert(tk.END, "\nMahad Istiqama Assistant: Sema kitu...\n")
        chatbox.see(tk.END)
        audio = recognizer.listen(source)
    try:
        text = recognizer.recognize_google(audio, language='sw')
        chatbox.insert(tk.END, f"Wewe: {text}\n")
        chatbox.see(tk.END)
        return text
    except sr.UnknownValueError:
        msg = "Samahani, sikukusikia vizuri."
        chatbox.insert(tk.END, f"Mahad Istiqama Assistant: {msg}\n")
        speak_swahili(msg)
    except sr.RequestError:
        msg = "Huduma ya kutambua sauti haipatikani kwa sasa."
        chatbox.insert(tk.END, f"Mahad Istiqama Assistant: {msg}\n")
        speak_swahili(msg)
    return ""

def speak_swahili(text):
    try:
        tts = gTTS(text=text, lang='sw')
        filename = "response.mp3"
        tts.save(filename)
        playsound(filename)
        os.remove(filename)
    except Exception as e:
        print("Error playing audio:", e)

def get_response(text):
    text = text.lower()
    responses = {
        "hujambo": "Karibu kwa Msaidizi wa Matibabu...",
        "habari": "Karibu kwa Msaidizi wa Matibabu...",
        "jina lako": "Mimi ni Msaidizi wa Matibabu wa AI...",
        "homa": "Homa ni hali ambapo joto la mwili linapanda zaidi ya kawaida, na mara nyingi ni dalili ya mwili kupambana na maambukizi.Inaweza kuwa ni kutokana na mafua, maambukizi ya bakteria, virusi, au malaria Ikiwa homa itaendelea kwa zaidi ya siku mbili au itaambatana na dalili nyingine kama kutapika au kifafa, tafadhali nenda hospitali",
        "kichwa": "Maumivu ya kichwa yanaweza kuwa kutokana na sababu nyingi kama vile msongo wa mawazo (stress), usingizi hafifu, shinikizo la damu, Lakini ikiwa maumivu ni makali sana, mara kwa mara, au yanaambatana na kichefuchefu, kizunguzungu au matatizo ya kuona, tafadhali mtembelee daktari",
        "tumbo": "Maumivu ya tumbo yanaweza kusababishwa na vyakula visivyofaa, gesi, vidonda vya tumbo, au maambukizi. Epuka kula vyakula vyenye mafuta mengi au viungo vikali. Ikiwa maumivu ni makali au yanadumu kwa muda mrefu, pata ushauri wa daktari.",
        "kikohozi": "Kikohozi kinaweza kuwa kikavu (hakina makohozi) au chenye makohozi, Ikiwa kikohozi kitaendelea kwa zaidi ya wiki mbili, au kimeambatana na damu, tafadhali mtembelee daktari.",
        "msongo": "Msongo wa mawazo unaweza kuathiri afya ya mwili na akili. Dalili hujumuisha huzuni, hasira, kuchoka haraka, au kukosa usingizi, Jaribu kufanya mazoezi, kutafakari (meditation), au kuzungumza na mtu unayemwamini, Kumbuka, si vibaya kuomba msaada. Afya ya akili ni muhimu kama afya ya mwili.",
        "moyo": "Maumivu wakati wa hedhi ni hali ya kawaida...",
        "kupumua": "Shida ya kupumua ni dalili mbaya inayoweza kuashiria matatizo ya mapafu, pumu (asthma), au hata maambukizi kama COVID-19 au nimonia. Kama unahisi kubanwa kwa kifua, pumzi fupi, au unapumua kwa shida, tafadhali pata msaada wa haraka hospitalini. Kwa wenye pumu, hakikisha una dawa ya *inhaler* kama *Ventolin*. Usisubiri hali iwe mbaya.",
        "mtoto ana homa": "Watoto wanapopata homa, ni muhimu kuchukua hatua haraka, Tumia dawa za kushusha homa kama *Calpol* (paracetamol kwa watoto) au *Ibuprofen* kwa kipimo sahihi kulingana na umri na uzito. Ikiwa mtoto ana degedege, analala sana au hachangamki, mpeleke hospitali mara moja.",
        "kwaheri": "Kwaheri! Ahsante kwa kutumia Msaidizi...",
        "toka": "Kwaheri! Ahsante kwa kutumia Msaidizi..."
    }
    for key in responses:
        if key in text:
            return responses[key]
    return "Samahani, sielewi swali hilo. Tafadhali jaribu tena."

def respond_from_input(user_input):
    if not user_input.strip():
        return
    response = get_response(user_input)
    chatbox.insert(tk.END, f"Mahad Istiqama Assistant: {response}\n")
    chatbox.see(tk.END)
    threading.Thread(target=speak_swahili, args=(response,)).start()

# ==== MAIN UI ====

def open_main_window():
    global chatbox
    root = tk.Tk()
    root.title("🤖 Msaidizi wa Matibabu (Kiswahili)")
    root.geometry("700x620")
    root.configure(bg="#eaf2f8")

    tk.Label(root, text="Msaidizi wa Matibabu kwa Kiswahili",
             font=("Helvetica", 18, "bold"), fg="#21618c", bg="#eaf2f8").pack(pady=10)
    tk.Label(root, text="Maahad Istiqama, Tunguu-Zanzibar",
             font=("Helvetica", 14, "bold"), fg="#21618c", bg="#eaf2f8").pack(pady=10)
    tk.Label(root, text="Science Show - 2025",
             font=("Helvetica", 12, "bold"), fg="#21618c", bg="#eaf2f8").pack(pady=10)

    chat_frame = tk.Frame(root, bg="#ffffff", bd=2, relief=tk.RIDGE)
    chat_frame.pack(padx=20, pady=10, fill=tk.BOTH, expand=True)

    chat_scrollbar = tk.Scrollbar(chat_frame)
    chat_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    chatbox = tk.Text(chat_frame, wrap=tk.WORD, font=("Arial", 11),
                      yscrollcommand=chat_scrollbar.set, bg="#fefefe", fg="#1a1a1a")
    chatbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    chat_scrollbar.config(command=chatbox.yview)

    control_frame = tk.Frame(root, bg="#eaf2f8")
    control_frame.pack(pady=10)

    tk.Button(control_frame, text="🎤 Zungumza", font=("Arial", 11, "bold"), bg="#5dade2",
              fg="white", width=15, command=lambda: respond_from_input(listen_in_swahili())).grid(row=0, column=0, padx=10)

    user_entry = tk.Entry(control_frame, font=("Arial", 11), width=40)
    user_entry.grid(row=0, column=1, padx=10)

    tk.Button(control_frame, text="Tuma", font=("Arial", 11, "bold"), bg="#58d68d",
              fg="white", width=10,
              command=lambda: respond_from_input(user_entry.get())).grid(row=0, column=2)

    symptom_frame = tk.Frame(root, bg="#eaf2f8")
    symptom_frame.pack(pady=15)

    tk.Label(symptom_frame, text="📌 Chagua Dalili: ", font=("Arial", 12, "bold"),
             bg="#eaf2f8", fg="#2c3e50").grid(row=0, columnspan=4, pady=(0, 10))

    symptoms = [
        ("Homa", "homa"), ("Tumbo", "tumbo"), ("Kichwa", "kichwa"),
        ("Kikohozi", "kikohozi"), ("Msongo", "msongo"),
        ("Moyo", "hedhi"), ("Kupumua", "kupumua"),
        ("Mtoto Ana Homa", "mtoto ana homa")
    ]

    for i, (label, command) in enumerate(symptoms):
        tk.Button(symptom_frame, text=label, width=20, font=("Arial", 10),
                  bg="#aed6f1", fg="#1b2631",
                  command=lambda cmd=command: respond_from_input(cmd)).grid(row=i//2+1, column=i%2, padx=10, pady=5)

    root.mainloop()

# Start App
open_main_window()

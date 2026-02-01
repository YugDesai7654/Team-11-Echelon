# Video Submission Script - Multi-Modal Misinformation Detection System
## Team 11 - Echelon | Total Duration: 7 Minutes

---

## PART 1: IMAGE ANALYSIS DEMO (4-5 Minutes)

### Opening Introduction (30 seconds)

**[Show: Home screen of the application]**

> "Hello everyone! Welcome to our demo of the Multi-Modal Misinformation Detection System, built by Team 11 Echelon.
>
> Our system uses AI to check if news claims are true or false. It works with both images and videos. Today, we will show you how it catches fake news in real-time.
>
> Our pipeline has 7 key parts, called deliverables. Let us walk you through each one."

---

### D1: Multi-Modal Input Handling (45 seconds)

**[Show: Upload Evidence section and Claim to Verify box]**

> "First, let us talk about D1 - Multi-Modal Input Handling.
>
> Our system can take different types of input. You can upload an image, a video, or just enter text. The system handles all of these smoothly.
>
> For this demo, I am uploading a flood image. The claim says: 'Breaking news: heavy flood in Surat, today.'
>
> Notice how the system shows 'Ready for analysis' once the image is uploaded. The file name, size, and status are all displayed clearly.
>
> This stage prepares the data for all the other stages that come next."

---

### D2: Cross-Modal Analysis (1 minute)

**[Click: RUN FULL PIPELINE ANALYSIS button]**

**[Show: CLIP Consistency score of 0.32]**

> "Now comes D2 - Cross-Modal Inconsistency Detection.
>
> This uses a technology called CLIP, made by OpenAI. CLIP checks if the text and image actually match each other.
>
> As you can see, the CLIP Consistency score is 0.32 out of 1.00. This is a low score. It means the caption and image do not align well.
>
> A score close to 1.00 means the text and image match perfectly. A low score like 0.32 raises a red flag - something might be wrong with this claim."

---

### D3: Out-of-Context Detection (45 seconds)

**[Show: Web search results and evidence in the explanation]**

> "D3 is about finding Out-of-Context media.
>
> The system searches the web to check if this image has been used before in a different context.
>
> Look at the results! The web search found that this exact flood image has been used many times since 2006. It has been linked to Mumbai, Changsha in China, and other places.
>
> So the image is real - it does show a flood. But the claim that this is 'Surat, today' is completely false. This is what we call out-of-context misinformation."

---

### D4: Synthetic Media Detection (45 seconds)

**[Show: AI Text Probability 89.1% and Deepfake Probability 36.6%]**

> "D4 detects AI-generated or synthetic media.
>
> We check two things here. First, is the text written by AI? Second, is the image a deepfake?
>
> The AI Text Probability is 89.1%. This suggests the caption might be AI-generated.
>
> The Deepfake Probability is 36.6%. This is low, which means the image is likely a real photograph, not AI-created.
>
> This confirms our finding - the image is real, but it is being used with a fake caption."

---

### D5: Explanation Generation (30 seconds)

**[Show: The detailed explanation in the OUT-OF-CONTEXT verdict box]**

> "D5 generates a clear explanation of our findings.
>
> The system uses AI to write a human-readable summary. Look at this explanation:
>
> 'The image accurately depicts a heavy flood, which aligns with the visual content. However, the claim that this is breaking news occurring in Surat today is false. Web search results show this image has been used since 2006.'
>
> This makes it easy for anyone to understand why the content is marked as misinformation."

---

### D6: Robustness Check (30 seconds)

**[Show: Pipeline Analysis Details section - Robustness Score]**

> "D6 protects our system against adversarial attacks - tricks that bad actors use to fool AI.
>
> For text, we check for three types of attacks:
> - Hidden zero-width characters that humans cannot see
> - Homoglyph attacks where letters from different alphabets look the same, like Cyrillic 'е' and Latin 'e'
> - Prompt injection attempts that try to confuse the AI
>
> For images, we check if the image has been tampered with, like having a uniform color channel.
>
> The system gives a Robustness Score. If it drops below 50% or finds more than 2 problems, the content is flagged as suspicious."

---

### D7: Final Verdict and Trust Score (30 seconds)

**[Show: OUT-OF-CONTEXT verdict with Trust Score of 10]**

> "Finally, D7 brings everything together.
>
> The system gives a final verdict: OUT-OF-CONTEXT.
>
> The Trust Score is only 10 out of 100. This means the content should not be trusted.
>
> All scores are shown clearly on the screen - CLIP Consistency, AI Text Probability, and Deepfake Probability. The user can see exactly why this content was flagged."

---

## PART 2: CHATBOT DEMO (1 Minute)

### Evidence-Based Chat Assistant (1 minute)

**[Show: Chat interface with the question "what evidence was found?"]**

> "Now let me show you our chatbot feature.
>
> After the analysis is done, users can ask questions about the results. This makes the system interactive and user-friendly.
>
> I am asking: 'What evidence was found?'
>
> The chatbot replies with all the key evidence. It explains:
> - The web search showed the 'Breaking News Today' claim is not supported
> - The image is old, dating back to 2006
> - It has been used for floods in Mumbai, Changsha, and other places
> - The image is real but the context is completely false
>
> Users can ask follow-up questions and get detailed answers based on the analysis. This helps people understand exactly why content is marked as misinformation."

---

## PART 3: VIDEO ANALYSIS DEMO (1-1.5 Minutes)

### Video Input and Deepfake Detection (1-1.5 minutes)

**[Show: Upload a video file - Govinda's Avatar 3 claim video]**

> "Our system also works with videos. Let me show you.
>
> I am uploading a video where someone claims that actor Govinda appeared in Avatar 3.
>
> The system processes the video frame by frame. It extracts key frames and analyzes each one."

**[Show: Video analysis results with 100% Deepfake Probability]**

> "Look at the results! The Deepfake Probability is 100%. This clearly shows the video is manipulated.
>
> The final verdict is UNVERIFIED with a Trust Score of 0.
>
> The CLIP Consistency is 1.00 because the text and video content do match - the video does show what the claim says. But our deepfake detector caught that the video itself is fake.
>
> This shows how our multi-modal approach catches different types of misinformation. Sometimes the content matches the claim, but the content itself is fabricated. Our system catches that too."

---

## CLOSING (15-30 seconds)

**[Show: Full application interface]**

> "To summarize, our Multi-Modal Misinformation Detection System:
>
> - Handles images, videos, and text
> - Uses CLIP for cross-modal analysis
> - Detects out-of-context media through web search
> - Identifies AI-generated and deepfake content
> - Provides clear explanations
> - Includes an interactive chatbot
>
> Thank you for watching! This was Team 11 Echelon."

---

## Quick Reference - Timing Summary

| Section | Duration | Deliverable |
|---------|----------|-------------|
| Introduction | 30 sec | Overview |
| D1: Input Handling | 45 sec | Multi-modal input |
| D2: Cross-Modal Analysis | 1 min | CLIP consistency |
| D3: Context Detection | 45 sec | Web search, OOC |
| D4: Synthetic Detection | 45 sec | Deepfake, AI text |
| D5: Explanation | 30 sec | Natural language |
| D6: Robustness | 30 sec | Adversarial checks |
| D7: Final Verdict | 30 sec | Trust score |
| **Image Total** | **~5 min** | - |
| Chatbot Demo | 1 min | Q&A feature |
| Video Analysis | 1-1.5 min | Video deepfake |
| Closing | 15-30 sec | Summary |
| **Total** | **~7 min** | All D1-D7 |

---

## Recording Tips

1. **Speak slowly and clearly** - Viewers need time to read the screen
2. **Pause briefly** after clicking buttons to show loading states
3. **Highlight important numbers** by pointing or zooming
4. **Keep cursor movements smooth** and purposeful
5. **Have all test files ready** before recording (flood image + Govinda video)
6. **Test the chatbot questions** beforehand to ensure good responses

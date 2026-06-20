from fpdf import FPDF
from datetime import datetime
import io

class PDFReport(FPDF):
    def header(self):
        # Logo could be added here if available
        # self.image('logo-color.png', 10, 8, 33)
        self.set_font('helvetica', 'B', 20)
        self.set_text_color(41, 128, 185) # Blue title
        self.cell(0, 15, 'LucidEpoch Clinical Sleep Assessment', ln=True, align='C')
        self.set_font('helvetica', 'I', 10)
        self.set_text_color(100, 100, 100)
        self.cell(0, 5, f'Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M")}', ln=True, align='C')
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font('helvetica', 'I', 8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 10, f'Page {self.page_no()}/{{nb}} - LucidEpoch AI Diagnostics', 0, 0, 'C')

def generate_pdf_report(patient_info, risk_scores, explanations_text, max_score, risk_names):
    pdf = PDFReport()
    pdf.alias_nb_pages()
    pdf.add_page()
    
    # Patient Metrics Section
    pdf.set_font('helvetica', 'B', 14)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(0, 10, 'Patient Metrics', ln=True)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(5)
    
    pdf.set_font('helvetica', '', 11)
    
    # Two column layout for metrics
    metrics = [
        f"Patient ID: {patient_info.get('patient_id', 'N/A')}",
        f"Age: {patient_info['age']} years",
        f"Gender: {patient_info['gender']}",
        f"Sleep Duration: {patient_info['sleep_duration']} hrs",
        f"Sleep Efficiency: {int(patient_info['sleep_efficiency']*100)}%",
        f"REM Sleep: {patient_info['rem_sleep']}%",
        f"Deep Sleep: {patient_info['deep_sleep']}%",
        f"Light Sleep: {patient_info['light_sleep']}%"
    ]
    
    col1_y = pdf.get_y()
    for i in range(0, len(metrics), 2):
        pdf.set_x(15)
        pdf.cell(90, 8, metrics[i], border=0)
        if i+1 < len(metrics):
            pdf.set_x(105)
            pdf.cell(90, 8, metrics[i+1], border=0)
        pdf.ln(8)
        
    pdf.ln(10)
    
    # Diagnostic Risk Predictions Section
    pdf.set_font('helvetica', 'B', 14)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(0, 10, 'Diagnostic Risk Predictions', ln=True)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(5)
    
    for target_col, info in risk_scores.items():
        pdf.set_font('helvetica', 'B', 12)
        
        # Color coding risks
        if info['score'] == 0:
            pdf.set_text_color(46, 204, 113) # Green
        elif info['score'] == 1:
            pdf.set_text_color(243, 156, 18) # Orange
        else:
            pdf.set_text_color(231, 76, 60) # Red
            
        disorder_name = target_col.replace('_', ' ').upper()
        pdf.cell(0, 8, f"{disorder_name}: {info['label']}", ln=True)
        
        pdf.set_text_color(0, 0, 0)
        if info['score'] > 0 and explanations_text.get(target_col):
            pdf.set_font('helvetica', 'I', 11)
            clean_exp = explanations_text[target_col].replace('**', '')
            pdf.set_x(15)
            pdf.multi_cell(0, 6, f"Reason: {clean_exp}")
        pdf.ln(4)
        
    pdf.ln(5)
    
    # Recommendation Section
    pdf.set_font('helvetica', 'B', 14)
    pdf.cell(0, 10, 'Clinical Recommendation', ln=True)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(5)
    
    pdf.set_font('helvetica', '', 11)
    if max_score > 0:
        pdf.set_fill_color(255, 235, 238) # Light red background
        pdf.multi_cell(0, 8, f"MEDICAL CONSULTATION RECOMMENDED.\n\nBased on the patient's metrics, the most prominent risk factor is {risk_names.replace('**', '')}. We strongly recommend consulting with a doctor or sleep specialist for a professional diagnosis and guidance.", border=1, fill=True, align='L')
    else:
        pdf.set_fill_color(232, 245, 233) # Light green background
        pdf.multi_cell(0, 8, "Predicted sleep disorder risks are low. No immediate medical consultation is required based on these metrics. Encourage continued good sleep hygiene.", border=1, fill=True, align='L')
        
    return bytes(pdf.output())

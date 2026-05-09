#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Groq + RAG Chatbot for Graphene Membrane Desalination
Expert system combining retrieval-augmented generation with Groq API
"""

from __future__ import annotations
from dataclasses import dataclass
import os
import sys
import re
from typing import List, Dict, Optional, Tuple
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from groq import Groq
import pickle

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@dataclass(frozen=True)
class ChatbotResponse:
    text: str
    blocked: bool = False


WATER_DOMAIN_SYSTEM_RULES = (
    "You are an assistant embedded in a web tool for seawater desalination research "
    "focused on single-layer graphene nanopore membranes. "
    "Only answer questions related to water desalination, membrane science, "
    "nanoporous graphene, and interpreting model outputs. "
    "If the user asks about unrelated topics (sports, politics, etc.), refuse briefly "
    "and steer back to desalination/membranes. "
    "You must support Arabic, French, and English. Answer in the same language as the user."
)


class GrapheneRAGChatbot:
    """RAG-based chatbot specialized in graphene membrane desalination"""
    
    def __init__(self):
        """Initialize RAG chatbot"""
        self.groq_client = None
        self.embeddings_model = None
        self.faiss_index = None
        self.knowledge_chunks = []
        self.simulation_models = {}
        
        # Initialize components
        self._initialize_groq()
        self._initialize_embeddings()
        self._load_knowledge_base()
        self._build_faiss_index()
        self._load_simulation_models()
    
    def _initialize_groq(self):
        """Initialize Groq client"""
        try:
            # Get API key from environment or secrets
            api_key = os.environ.get("GROQ_API_KEY")
            if not api_key:
                # Try to get from Streamlit secrets
                try:
                    import streamlit as st
                    api_key = st.secrets.get("GROQ_API_KEY")
                except:
                    pass
            
            if not api_key:
                raise ValueError("GROQ_API_KEY not found")
            
            self.groq_client = Groq(api_key=api_key)
            print("✅ Groq client initialized successfully")
            
        except Exception as e:
            print(f"❌ Error initializing Groq: {e}")
            self.groq_client = None
    
    def _initialize_embeddings(self):
        """Initialize sentence transformer model"""
        try:
            # Use lightweight multilingual model
            self.embeddings_model = SentenceTransformer('all-MiniLM-L6-v2')
            print("✅ Embeddings model loaded successfully")
        except Exception as e:
            print(f"❌ Error loading embeddings model: {e}")
            self.embeddings_model = None
    
    def _load_knowledge_base(self):
        """Load and chunk knowledge base"""
        try:
            # Load knowledge base text
            kb_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'knowledge_base.txt')
            with open(kb_path, 'r', encoding='utf-8') as f:
                kb_text = f.read()
            
            # Create chunks (sections separated by ===)
            chunks = []
            sections = re.split(r'={3,}', kb_text)
            
            for section in sections:
                if section.strip():
                    # Further split into paragraphs
                    paragraphs = section.strip().split('\n\n')
                    for para in paragraphs:
                        if len(para.strip()) > 50:  # Only keep substantial paragraphs
                            chunks.append(para.strip())
            
            # Add Excel data if available
            try:
                excel_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'DATA ARTCLE 1 A 12 FINAL.xlsx')
                if os.path.exists(excel_path):
                    df = pd.read_excel(excel_path)
                    # Create descriptive chunks from data
                    for _, row in df.iterrows():
                        description = f"Simulation data: Geometry={row.get('Geometry', 'N/A')}, "
                        description += f"Pore Area={row.get('Pore Area', 'N/A')} Å², "
                        description += f"Pressure={row.get('Applied Pressure', 'N/A')} MPa, "
                        description += f"Salt Rejection={row.get('Salt Rejection', 'N/A')}%, "
                        description += f"Water Flux={row.get('Water Flux', 'N/A')} molecules/ns"
                        chunks.append(description)
            except Exception as e:
                print(f"⚠️ Could not load Excel data: {e}")
            
            self.knowledge_chunks = chunks
            print(f"✅ Loaded {len(chunks)} knowledge chunks")
            
        except Exception as e:
            print(f"❌ Error loading knowledge base: {e}")
            self.knowledge_chunks = []
    
    def _build_faiss_index(self):
        """Build FAISS index for fast retrieval"""
        try:
            if not self.embeddings_model or not self.knowledge_chunks:
                return
            
            # Generate embeddings
            embeddings = self.embeddings_model.encode(
                self.knowledge_chunks, 
                convert_to_numpy=True,
                show_progress_bar=False
            )
            
            # Build FAISS index
            dimension = embeddings.shape[1]
            self.faiss_index = faiss.IndexFlatL2(dimension)
            self.faiss_index.add(embeddings.astype('float32'))
            
            print(f"✅ FAISS index built with {len(embeddings)} vectors")
            
        except Exception as e:
            print(f"❌ Error building FAISS index: {e}")
            self.faiss_index = None
    
    def _load_simulation_models(self):
        """Load CatBoost models for simulation"""
        try:
            # Try to load existing models
            model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
            
            # For now, we'll create dummy predictions
            # In production, load actual CatBoost models
            
            self.simulation_models = {
                'salt_rejection': None,  # Will be replaced with actual model
                'water_flux': None       # Will be replaced with actual model
            }
            
            print("✅ Simulation models initialized (placeholder)")
            
        except Exception as e:
            print(f"❌ Error loading simulation models: {e}")
            self.simulation_models = {}
    
    def _retrieve_relevant_chunks(self, query: str, top_k: int = 3) -> List[str]:
        """Retrieve most relevant knowledge chunks"""
        try:
            if not self.faiss_index or not self.embeddings_model:
                return []
            
            # Generate query embedding
            query_embedding = self.embeddings_model.encode([query], convert_to_numpy=True)
            
            # Search FAISS index
            distances, indices = self.faiss_index.search(
                query_embedding.astype('float32'), 
                min(top_k, len(self.knowledge_chunks))
            )
            
            # Return relevant chunks
            relevant_chunks = [self.knowledge_chunks[i] for i in indices[0]]
            return relevant_chunks
            
        except Exception as e:
            print(f"❌ Error retrieving chunks: {e}")
            return []
    
    def _extract_simulation_parameters(self, message: str) -> Optional[Dict]:
        """Extract simulation parameters from user message"""
        try:
            # Define parameter patterns
            patterns = {
                'pore_area': r'pore area[:\s]*([0-9]+(?:\.[0-9]*)?)\s*Å²',
                'pressure': r'pressure[:\s]*([0-9]+(?:\.[0-9]*)?)\s*MPa',
                'concentration': r'concentration[:\s]*([0-9]+(?:\.[0-9]*)?)\s*ppm',
                'temperature': r'temperature[:\s]*([0-9]+(?:\.[0-9]*)?)\s*°C',
                'porosity': r'porosity[:\s]*([0-9]+(?:\.[0-9]*)?)\s*%',
                'geometry': r'geometry[:\s]*(hexagonal|circular|triangular|rhombic|ozark)',
                'chemistry': r'chemistry[:\s]*(H|OH|N|NH|NO|FH|SiH2|Si\(OH\)2|Pristine|COO⁻)'
            }
            
            params = {}
            for param, pattern in patterns.items():
                match = re.search(pattern, message, re.IGNORECASE)
                if match:
                    if param in ['geometry', 'chemistry']:
                        params[param] = match.group(1)
                    else:
                        params[param] = float(match.group(1))
            
            return params if params else None
            
        except Exception as e:
            print(f"❌ Error extracting parameters: {e}")
            return None
    
    def _simulate_with_parameters(self, params: Dict) -> Optional[Dict]:
        """Run simulation with extracted parameters"""
        try:
            # For now, use placeholder calculations
            # In production, use actual CatBoost models
            
            # Default values
            defaults = {
                'pore_area': 30.0,
                'pressure': 100.0,
                'concentration': 35000.0,
                'temperature': 27.0,
                'porosity': 5.0,
                'geometry': 'hexagonal',
                'chemistry': 'H'
            }
            
            # Merge with extracted parameters
            sim_params = {**defaults, **params}
            
            # Placeholder calculations (replace with actual model predictions)
            pore_area = sim_params['pore_area']
            pressure = sim_params['pressure']
            
            # Simple heuristic calculations
            if pore_area < 24:
                salt_rejection = 99.5 + np.random.normal(0, 0.3)
                salt_rejection = min(100.0, max(95.0, salt_rejection))
            else:
                salt_rejection = max(50.0, 100.0 - (pore_area - 24) * 2)
            
            water_flux = (pore_area * pressure / 100.0) + np.random.normal(0, 10)
            water_flux = max(0.1, water_flux)
            
            return {
                'salt_rejection': round(salt_rejection, 2),
                'water_flux': round(water_flux, 2),
                'parameters': sim_params
            }
            
        except Exception as e:
            print(f"❌ Error in simulation: {e}")
            return None
    
    def _create_rag_prompt(self, query: str, context: List[str]) -> str:
        """Create RAG prompt for Groq"""
        context_text = "\n\n".join(context[:3])  # Use top 3 chunks
        
        prompt = f"""You are an expert in graphene membrane desalination and water treatment. 
You have access to specialized knowledge about single-layer nanoporous graphene (SLNG) membranes.

CONTEXT FROM KNOWLEDGE BASE:
{context_text}

USER QUESTION: {query}

Instructions:
1. Answer based ONLY on the provided context
2. Be specific and technical when appropriate
3. If user asks for simulation results, provide the calculated values
4. Answer in the same language as the question (Arabic/French/English)
5. Be helpful and professional
6. If you cannot find the answer in the context, say so politely

Answer:"""
        
        return prompt
    
    def answer_message(self, message: str) -> str:
        """Main method to answer user messages"""
        try:
            if not self.groq_client:
                return "❌ Groq client not initialized. Please check API key."
            
            # Check if user wants simulation
            sim_params = self._extract_simulation_parameters(message)
            simulation_result = None
            
            if sim_params:
                simulation_result = self._simulate_with_parameters(sim_params)
            
            # Retrieve relevant context
            relevant_chunks = self._retrieve_relevant_chunks(message)
            
            # Create prompt
            prompt = self._create_rag_prompt(message, relevant_chunks)
            
            # Add simulation results to context if available
            if simulation_result:
                prompt += f"\n\nSIMULATION RESULTS: {simulation_result}"
            
            # Call Groq API
            response = self.groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",  # Use Llama 3.3 70B Versatile
                messages=[
                    {"role": "system", "content": "You are an expert in graphene membrane desalination."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1000,
                temperature=0.1
            )
            
            answer = response.choices[0].message.content
            
            # Add simulation results if available
            if simulation_result:
                sim_text = f"\n\n🔬 **Simulation Results:**\n"
                sim_text += f"• Salt Rejection: {simulation_result['salt_rejection']}%\n"
                sim_text += f"• Water Flux: {simulation_result['water_flux']} molecules/ns\n"
                sim_text += f"• Parameters used: {simulation_result['parameters']}"
                answer += sim_text
            
            return answer
            
        except Exception as e:
            print(f"❌ Error answering message: {e}")
            return f"❌ Sorry, I encountered an error: {str(e)}"

# Global instance
_chatbot_instance = None

def get_chatbot():
    """Get or create chatbot instance"""
    global _chatbot_instance
    if _chatbot_instance is None:
        _chatbot_instance = GrapheneRAGChatbot()
    return _chatbot_instance

def answer_user_message(message: str, api_key: str = None, model: str = "llama3-70b-8192") -> ChatbotResponse:
    """
    Groq + RAG backed chatbot.
    Falls back to local rule-based response if API is not available.
    """
    # Check for blocked keywords
    lower = message.lower().strip()
    blocked_keywords = ["football", "soccer", "politic", "election", "gym", "workout", "basketball"]
    if any(k in lower for k in blocked_keywords):
        return ChatbotResponse(
            text="I can only help with seawater desalination and graphene nanopore membrane topics inside this tool.",
            blocked=True,
        )
    
    try:
        # Try to use the RAG chatbot
        chatbot = get_chatbot()
        answer = chatbot.answer_message(message)
        return ChatbotResponse(text=answer, blocked=False)
        
    except Exception as e:
        print(f"RAG chatbot error: {e}")
        # Fallback response
        return ChatbotResponse(
            text=(
                "🤖 **AI Assistant**\n\n"
                "I'm here to help with graphene membrane desalination questions! "
                "You can ask me about:\n"
                "• Salt rejection and water flux\n"
                "• Membrane properties and geometry\n"
                "• Simulation parameters\n"
                "• Graphene nanopore technology\n\n"
                "Please try rephrasing your question about desalination or membrane science."
            ),
            blocked=False,
        )

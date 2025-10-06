"""
Command-line interface for PersonaBot
Interactive chat interface with personality visualization
"""

import sys
import os
import argparse
import json
from typing import Optional, Dict, Any
from datetime import datetime
import logging

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.personabot import PersonaBot
from config.settings import settings
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Prompt, Confirm
from rich.layout import Layout
from rich.live import Live
from rich.align import Align

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PersonaBotCLI:
    """Command-line interface for PersonaBot"""
    
    def __init__(self):
        """Initialize CLI"""
        self.console = Console()
        self.bot = None
        self.session_file = None
        
        # CLI state
        self.running = True
        self.show_debug = False
        
    def run(self):
        """Main CLI loop"""
        self.console.print(Panel.fit(
            "[bold blue]🤖 PersonaBot - Advanced Conversational AI[/bold blue]\n"
            "[dim]Dynamic personality adaptation using Reinforcement Learning[/dim]",
            border_style="blue"
        ))
        
        # Parse command line arguments
        parser = argparse.ArgumentParser(description="PersonaBot CLI")
        parser.add_argument("--personality", type=str, help="Path to personality file")
        parser.add_argument("--session", type=str, help="Path to session file")
        parser.add_argument("--no-rl", action="store_true", help="Disable reinforcement learning")
        parser.add_argument("--debug", action="store_true", help="Enable debug mode")
        
        args = parser.parse_args()
        
        # Initialize bot
        self._initialize_bot(args)
        
        # Main interaction loop
        while self.running:
            try:
                self._show_menu()
                choice = Prompt.ask("Select option", choices=["1", "2", "3", "4", "5", "6", "7", "8", "9", "q"])
                
                if choice == "1":
                    self._start_chat()
                elif choice == "2":
                    self._show_personality()
                elif choice == "3":
                    self._show_performance()
                elif choice == "4":
                    self._save_session()
                elif choice == "5":
                    self._load_session()
                elif choice == "6":
                    self._export_personality()
                elif choice == "7":
                    self._import_personality()
                elif choice == "8":
                    self._reset_conversation()
                elif choice == "9":
                    self._show_debug_info()
                elif choice == "q":
                    self._quit()
                    
            except KeyboardInterrupt:
                self._quit()
            except Exception as e:
                self.console.print(f"[red]Error: {e}[/red]")
                logger.error(f"CLI error: {e}")
    
    def _initialize_bot(self, args):
        """Initialize the PersonaBot instance"""
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console
        ) as progress:
            task = progress.add_task("Initializing PersonaBot...", total=None)
            
            try:
                # Load personality if specified
                initial_personality = None
                if args.personality and os.path.exists(args.personality):
                    with open(args.personality, 'r') as f:
                        personality_data = json.load(f)
                        initial_personality = personality_data.get('current_state', {}).get('traits', {})
                
                # Initialize bot
                self.bot = PersonaBot(
                    initial_personality=initial_personality,
                    enable_rl=not args.no_rl
                )
                
                # Load session if specified
                if args.session and os.path.exists(args.session):
                    if self.bot.load_session(args.session):
                        self.session_file = args.session
                        self.console.print(f"[green]Session loaded from {args.session}[/green]")
                
                # Set debug mode
                if args.debug:
                    self.show_debug = True
                    logging.getLogger().setLevel(logging.DEBUG)
                
                progress.update(task, description="PersonaBot initialized successfully!")
                
            except Exception as e:
                self.console.print(f"[red]Failed to initialize PersonaBot: {e}[/red]")
                sys.exit(1)
    
    def _show_menu(self):
        """Display main menu"""
        self.console.print("\n")
        menu = Table(title="PersonaBot Menu", show_header=False, box=None)
        menu.add_column("Option", style="cyan")
        menu.add_column("Description", style="white")
        
        menu.add_row("1", "Start Chat")
        menu.add_row("2", "Show Personality")
        menu.add_row("3", "Show Performance")
        menu.add_row("4", "Save Session")
        menu.add_row("5", "Load Session")
        menu.add_row("6", "Export Personality")
        menu.add_row("7", "Import Personality")
        menu.add_row("8", "Reset Conversation")
        menu.add_row("9", "Debug Info")
        menu.add_row("q", "Quit")
        
        self.console.print(menu)
    
    def _start_chat(self):
        """Start interactive chat session"""
        self.console.print("\n[bold green]Starting chat session...[/bold green]")
        
        # Start conversation
        welcome = self.bot.start_conversation()
        self.console.print(f"[blue]🤖 PersonaBot:[/blue] {welcome}")
        
        # Chat loop
        while True:
            try:
                user_input = Prompt.ask("\n[green]You[/green]")
                
                if user_input.lower() in ['quit', 'exit', 'bye', 'q']:
                    self.console.print("[yellow]Ending chat session...[/yellow]")
                    break
                
                if user_input.lower() in ['help', 'h']:
                    self._show_chat_help()
                    continue
                
                if user_input.lower() in ['personality', 'p']:
                    self._show_current_personality()
                    continue
                
                if user_input.lower() in ['performance', 'perf']:
                    self._show_current_performance()
                    continue
                
                # Generate response
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    console=self.console,
                    transient=True
                ) as progress:
                    task = progress.add_task("Generating response...", total=None)
                    response = self.bot.chat(user_input)
                
                # Display response
                self.console.print(f"[blue]🤖 PersonaBot:[/blue] {response}")
                
                # Show debug info if enabled
                if self.show_debug:
                    self._show_debug_response()
                
            except KeyboardInterrupt:
                self.console.print("\n[yellow]Chat interrupted. Returning to menu...[/yellow]")
                break
            except Exception as e:
                self.console.print(f"[red]Error in chat: {e}[/red]")
    
    def _show_chat_help(self):
        """Show chat help"""
        help_text = """
[bold]Chat Commands:[/bold]
• [cyan]help[/cyan] or [cyan]h[/cyan] - Show this help
• [cyan]personality[/cyan] or [cyan]p[/cyan] - Show current personality
• [cyan]performance[/cyan] or [cyan]perf[/cyan] - Show performance metrics
• [cyan]quit[/cyan], [cyan]exit[/cyan], [cyan]bye[/cyan], or [cyan]q[/cyan] - End chat session
        """
        self.console.print(Panel(help_text, title="Chat Help", border_style="cyan"))
    
    def _show_personality(self):
        """Display current personality"""
        personality_summary = self.bot.get_personality_summary()
        
        # Create personality table
        table = Table(title="Current Personality", show_header=True)
        table.add_column("Trait", style="cyan")
        table.add_column("Value", style="green")
        table.add_column("Level", style="yellow")
        
        for trait, value in personality_summary['current_traits'].items():
            if value > 0.7:
                level = "High"
                style = "green"
            elif value < 0.3:
                level = "Low"
                style = "red"
            else:
                level = "Medium"
                style = "yellow"
            
            table.add_row(trait.title(), f"{value:.2f}", f"[{style}]{level}[/{style}]")
        
        self.console.print(table)
        
        # Show metrics
        metrics_table = Table(title="Personality Metrics", show_header=True)
        metrics_table.add_column("Metric", style="cyan")
        metrics_table.add_column("Value", style="green")
        
        metrics_table.add_row("Stability", f"{personality_summary['stability']:.3f}")
        metrics_table.add_row("Adaptation Count", str(personality_summary['adaptation_count']))
        metrics_table.add_row("Drift", f"{personality_summary['drift']:.3f}")
        
        self.console.print(metrics_table)
        self.console.print(f"[dim]{personality_summary['summary']}[/dim]")
    
    def _show_performance(self):
        """Display performance metrics"""
        performance = self.bot.get_performance_summary()
        
        # Create performance table
        table = Table(title="Performance Summary", show_header=True)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Total Interactions", str(performance['total_interactions']))
        table.add_row("Average Sentiment", f"{performance['average_sentiment']:.3f}")
        table.add_row("Average Engagement", f"{performance['average_engagement']:.3f}")
        table.add_row("Personality Adaptations", str(performance['personality_adaptations']))
        table.add_row("Session Duration", performance['session_duration'])
        table.add_row("Conversation Length", str(performance['conversation_length']))
        table.add_row("RL Enabled", "Yes" if performance['rl_enabled'] else "No")
        
        self.console.print(table)
        
        # Show RL stats if available
        if performance['rl_stats']:
            rl_table = Table(title="Reinforcement Learning Stats", show_header=True)
            rl_table.add_column("Metric", style="cyan")
            rl_table.add_column("Value", style="green")
            
            rl_stats = performance['rl_stats']
            rl_table.add_row("Epsilon", f"{rl_stats['epsilon']:.3f}")
            rl_table.add_row("Memory Size", str(rl_stats['memory_size']))
            rl_table.add_row("Episodes", str(rl_stats['training_stats']['episodes']))
            rl_table.add_row("Average Reward", f"{rl_stats['training_stats']['average_reward']:.3f}")
            
            self.console.print(rl_table)
    
    def _save_session(self):
        """Save current session"""
        if not self.bot.conversation_history:
            self.console.print("[yellow]No conversation to save.[/yellow]")
            return
        
        # Suggest filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_filename = f"session_{timestamp}.json"
        
        filename = Prompt.ask("Enter filename", default=default_filename)
        
        if not filename.endswith('.json'):
            filename += '.json'
        
        filepath = os.path.join(settings.data.sessions_dir, filename)
        
        try:
            saved_path = self.bot.save_session(filepath)
            self.console.print(f"[green]Session saved to {saved_path}[/green]")
            self.session_file = saved_path
        except Exception as e:
            self.console.print(f"[red]Failed to save session: {e}[/red]")
    
    def _load_session(self):
        """Load a session"""
        available_sessions = self.bot.get_available_sessions()
        
        if not available_sessions:
            self.console.print("[yellow]No saved sessions found.[/yellow]")
            return
        
        # Show available sessions
        table = Table(title="Available Sessions", show_header=True)
        table.add_column("Index", style="cyan")
        table.add_column("Filename", style="green")
        table.add_column("Date", style="yellow")
        
        for i, session_path in enumerate(available_sessions[:10]):  # Show last 10
            filename = os.path.basename(session_path)
            date_str = filename.replace('session_', '').replace('.json', '')
            table.add_row(str(i + 1), filename, date_str)
        
        self.console.print(table)
        
        choice = Prompt.ask("Select session to load", choices=[str(i) for i in range(1, min(11, len(available_sessions) + 1))])
        
        try:
            selected_session = available_sessions[int(choice) - 1]
            if self.bot.load_session(selected_session):
                self.console.print(f"[green]Session loaded from {selected_session}[/green]")
                self.session_file = selected_session
            else:
                self.console.print("[red]Failed to load session.[/red]")
        except Exception as e:
            self.console.print(f"[red]Error loading session: {e}[/red]")
    
    def _export_personality(self):
        """Export current personality"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_filename = f"personality_{timestamp}.json"
        
        filename = Prompt.ask("Enter filename", default=default_filename)
        
        if not filename.endswith('.json'):
            filename += '.json'
        
        filepath = os.path.join(settings.data.models_dir, filename)
        
        try:
            self.bot.export_personality(filepath)
            self.console.print(f"[green]Personality exported to {filepath}[/green]")
        except Exception as e:
            self.console.print(f"[red]Failed to export personality: {e}[/red]")
    
    def _import_personality(self):
        """Import personality from file"""
        filename = Prompt.ask("Enter personality file path")
        
        if not os.path.exists(filename):
            self.console.print("[red]File not found.[/red]")
            return
        
        try:
            if self.bot.import_personality(filename):
                self.console.print(f"[green]Personality imported from {filename}[/green]")
            else:
                self.console.print("[red]Failed to import personality.[/red]")
        except Exception as e:
            self.console.print(f"[red]Error importing personality: {e}[/red]")
    
    def _reset_conversation(self):
        """Reset conversation"""
        if Confirm.ask("Are you sure you want to reset the conversation?"):
            self.bot.reset_conversation()
            self.console.print("[green]Conversation reset.[/green]")
    
    def _show_debug_info(self):
        """Show debug information"""
        model_info = self.bot.get_model_info()
        
        # NLP Engine info
        nlp_table = Table(title="NLP Engine Info", show_header=True)
        nlp_table.add_column("Property", style="cyan")
        nlp_table.add_column("Value", style="green")
        
        nlp_info = model_info['nlp_engine']
        for key, value in nlp_info.items():
            nlp_table.add_row(key, str(value))
        
        self.console.print(nlp_table)
        
        # Personality encoder info
        personality_table = Table(title="Personality Encoder Info", show_header=True)
        personality_table.add_column("Property", style="cyan")
        personality_table.add_column("Value", style="green")
        
        personality_info = model_info['personality_encoder']
        personality_table.add_row("Traits", ", ".join(personality_info['traits']))
        personality_table.add_row("Current Values", str(personality_info['current_values']))
        
        self.console.print(personality_table)
        
        # RL Agent info if available
        if model_info['rl_agent']:
            rl_table = Table(title="RL Agent Info", show_header=True)
            rl_table.add_column("Property", style="cyan")
            rl_table.add_column("Value", style="green")
            
            rl_info = model_info['rl_agent']
            for key, value in rl_info.items():
                if key != 'training_stats':  # Skip complex nested data
                    rl_table.add_row(key, str(value))
            
            self.console.print(rl_table)
    
    def _show_current_personality(self):
        """Show current personality during chat"""
        personality = self.bot.get_personality_summary()
        
        # Create compact personality display
        traits_str = []
        for trait, value in personality['current_traits'].items():
            if value > 0.7:
                traits_str.append(f"[green]{trait.title()}: {value:.1f}[/green]")
            elif value < 0.3:
                traits_str.append(f"[red]{trait.title()}: {value:.1f}[/red]")
            else:
                traits_str.append(f"[yellow]{trait.title()}: {value:.1f}[/yellow]")
        
        self.console.print(f"[dim]Current Personality: {', '.join(traits_str)}[/dim]")
    
    def _show_current_performance(self):
        """Show current performance during chat"""
        performance = self.bot.get_performance_summary()
        
        self.console.print(f"[dim]Interactions: {performance['total_interactions']} | "
                          f"Sentiment: {performance['average_sentiment']:.2f} | "
                          f"Engagement: {performance['average_engagement']:.2f}[/dim]")
    
    def _show_debug_response(self):
        """Show debug information for the last response"""
        if not self.bot.conversation_history:
            return
        
        last_message = self.bot.conversation_history[-1]
        if 'rewards' in last_message:
            rewards = last_message['rewards']
            
            rewards_str = []
            for key, value in rewards.items():
                if key != 'total':
                    rewards_str.append(f"{key}: {value:.3f}")
            
            self.console.print(f"[dim]Rewards: {', '.join(rewards_str)}[/dim]")
    
    def _quit(self):
        """Quit the application"""
        if self.bot.conversation_history:
            if Confirm.ask("Save session before quitting?"):
                self._save_session()
        
        self.console.print("[green]Goodbye![/green]")
        self.running = False

def main():
    """Main entry point"""
    cli = PersonaBotCLI()
    cli.run()

if __name__ == "__main__":
    main()

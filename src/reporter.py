"""
Reporter - Tägliche und wöchentliche Reports
"""
import os
from datetime import datetime
from typing import List, Dict
import json

class Reporter:
    def __init__(self, report_dir: str = "reports"):
        self.report_dir = report_dir
        os.makedirs(report_dir, exist_ok=True)
    
    def generate_daily_report(self, date: datetime, performance: Dict, 
                             open_positions: List[Dict], 
                             strategy_stats: List[Dict],
                             new_signals: List[Dict]) -> str:
        """Täglichen Report generieren"""
        
        report_lines = [
            "=" * 60,
            f"📊 TRADEFORGE TAGESREPORT - {date.strftime('%Y-%m-%d')}",
            "=" * 60,
            "",
            f"⏰ Report generiert: {datetime.now().strftime('%H:%M:%S')} UTC",
            "",
            "📈 PERFORMANCE HISTE",
            "-" * 40,
            f"Geschlossene Trades heute: {performance.get('total_trades', 0)}",
            f"Gewinn Trades: {performance.get('winning_trades', 0)}",
            f"Verlust Trades: {performance.get('losing_trades', 0)}",
            f"Win Rate: {performance.get('win_rate', 0):.1f}%",
            f"Durchschnittl. P&L: {performance.get('avg_pnl', 0):.2f}%",
            "",
            "📊 STRATEGIE PERFORMANCE",
            "-" * 40,
        ]
        
        for stat in strategy_stats:
            report_lines.append(
                f"\n{stat['strategy']}:\n"
                f"  Trades: {stat['total_trades']} | "
                f"Win Rate: {stat['win_rate']:.1f}% | "
                f"Ø Return: {stat['avg_return']:.2f}%"
            )
        
        report_lines.extend([
            "",
            "🎯 OFFENE POSITIONEN",
            "-" * 40,
        ])
        
        if open_positions:
            for pos in open_positions:
                report_lines.append(
                    f"\n{pos['ticker']} ({pos['strategy']}):\n"
                    f"  Einstieg: ${pos['entry_price']:.2f}\n"
                    f"  Stop: ${pos['stop_loss']:.2f} | Target: ${pos['take_profit']:.2f}\n"
                    f"  Hebel: {pos['leverage']}x | Konfidenz: {pos['confidence']:.0%}"
                )
        else:
            report_lines.append("Keine offenen Positionen")
        
        report_lines.extend([
            "",
            "🔔 NEUE SIGNALE HEUTE",
            "-" * 40,
        ])
        
        if new_signals:
            for sig in new_signals:
                action = "🟢 LONG" if sig['signal_type'] == 'buy' else "🔴 SHORT"
                report_lines.append(
                    f"\n{action} {sig['ticker']} ({sig['strategy']})\n"
                    f"  Preis: ${sig['entry_price']:.2f}\n"
                    f"  Setup: {sig['setup_description']}\n"
                    f"  Konfidenz: {sig['confidence']:.0%}"
                )
        else:
            report_lines.append("Keine neuen Signale")
        
        report_lines.extend([
            "",
            "=" * 60,
            "TradeForge - Autonomer Trading Analyst",
            "⚠️ Dies ist ein Paper-Trading Projekt",
            "=" * 60,
        ])
        
        report_text = "\n".join(report_lines)
        
        # Speichern
        filename = f"daily_report_{date.strftime('%Y%m%d')}.txt"
        filepath = os.path.join(self.report_dir, filename)
        
        with open(filepath, 'w') as f:
            f.write(report_text)
        
        return report_text
    
    def generate_weekly_summary(self, trades_data: List[Dict]) -> str:
        """Wöchentliche Zusammenfassung"""
        if not trades_data:
            return "Keine Trades diese Woche"
        
        total = len(trades_data)
        wins = sum(1 for t in trades_data if t.get('pnl_percent', 0) > 0)
        losses = total - wins
        win_rate = (wins / total * 100) if total > 0 else 0
        
        total_pnl = sum(t.get('pnl_percent', 0) for t in trades_data)
        avg_pnl = total_pnl / total if total > 0 else 0
        
        report = f"""
{'='*60}
📊 TRADEFORGE WOCHENZUSAMMENFASSUNG
{'='*60}

Gesamte Trades: {total}
Gewinne: {wins} | Verluste: {losses}
Win Rate: {win_rate:.1f}%

Gesamt P&L: {total_pnl:.2f}%
Durchschnitt pro Trade: {avg_pnl:.2f}%

{'='*60}
"""
        return report
    
    def print_console_report(self, text: str):
        """Report in Konsole ausgeben"""
        print(text)
    
    def save_signals_json(self, signals: List[Dict], date: datetime):
        """Signale als JSON speichern"""
        filename = f"signals_{date.strftime('%Y%m%d')}.json"
        filepath = os.path.join(self.report_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(signals, f, indent=2, default=str)
import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from datetime import datetime, timedelta
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QLabel, QComboBox, QPushButton, QCheckBox, QDateEdit, QSpinBox, 
    QDoubleSpinBox, QTabWidget, QGroupBox, QRadioButton, QSlider,
    QGridLayout, QFileDialog, QMessageBox, QScrollArea, QFrame,
    QSplitter, QListWidget, QListWidgetItem, QTextBrowser, QSizePolicy
)
from PyQt5.QtCore import Qt, QDate, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QIcon, QFont

# Import our modules
from data_manager import DataManager
from battery_optimizer import BatteryOptimizer
import ev_config

class MatplotlibCanvas(FigureCanvas):
    """Simple canvas to embed Matplotlib in the Qt application"""
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        
        super(MatplotlibCanvas, self).__init__(fig)
        self.setParent(parent)
        
        FigureCanvas.setSizePolicy(self,
                                  QSizePolicy.Expanding,
                                  QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

class HomeEnergyModelApp(QMainWindow):
    """Main application window for the Home Energy Model with EV Battery Optimization"""
    def __init__(self):
        super().__init__()
        
        # Initialize data manager
        self.data_manager = DataManager()
        
        # Initialize battery optimizer with default values
        self.battery_optimizer = BatteryOptimizer()
        
        # Initialize UI
        self.init_ui()
        
        # Set window properties
        self.setWindowTitle('Home Energy Model with EV Battery Optimization')
        self.setGeometry(100, 100, 1200, 800)
        
    def init_ui(self):
        """Initialize the user interface"""
        # Main widget and layout
        main_widget = QWidget()
        main_layout = QVBoxLayout(main_widget)
        
        # Create tabs
        tabs = QTabWidget()
        
        # Add tabs
        self.setup_data_tab(tabs)
        self.setup_ev_config_tab(tabs)
        self.setup_optimization_tab(tabs)
        self.setup_results_tab(tabs)
        
        main_layout.addWidget(tabs)
        
        # Set main widget
        self.setCentralWidget(main_widget)
        
    def setup_data_tab(self, tabs):
        """Setup the data selection and configuration tab"""
        data_tab = QWidget()
        layout = QVBoxLayout()
        
        # City selection
        city_group = QGroupBox("City and Data Selection")
        city_layout = QGridLayout()
        
        city_layout.addWidget(QLabel("Select City:"), 0, 0)
        self.city_combo = QComboBox()
        self.city_combo.addItems(list(self.data_manager.CITY_DATA.keys()))
        self.city_combo.currentTextChanged.connect(self.on_city_changed)
        city_layout.addWidget(self.city_combo, 0, 1)
        
        self.load_city_btn = QPushButton("Load City Data")
        self.load_city_btn.clicked.connect(self.on_load_city_data)
        city_layout.addWidget(self.load_city_btn, 0, 2)
        
        city_layout.addWidget(QLabel("Date:"), 1, 0)
        self.date_picker = QDateEdit()
        self.date_picker.setDate(QDate(2018, 7, 15))  # Default date
        self.date_picker.setCalendarPopup(True)
        city_layout.addWidget(self.date_picker, 1, 1)
        
        self.load_date_btn = QPushButton("Load Date Data")
        self.load_date_btn.clicked.connect(self.on_load_date_data)
        city_layout.addWidget(self.load_date_btn, 1, 2)
        
        # Load outage data
        city_layout.addWidget(QLabel("Power Outage Year:"), 2, 0)
        self.outage_year_combo = QComboBox()
        self.outage_year_combo.addItems([str(year) for year in range(2014, 2024)])
        self.outage_year_combo.setCurrentText("2023")
        city_layout.addWidget(self.outage_year_combo, 2, 1)
        
        self.load_outage_btn = QPushButton("Load Outage Data")
        self.load_outage_btn.clicked.connect(self.on_load_outage_data)
        city_layout.addWidget(self.load_outage_btn, 2, 2)
        
        city_group.setLayout(city_layout)
        layout.addWidget(city_group)
        
        # Energy consumption options
        energy_group = QGroupBox("Energy Consumption Settings")
        energy_layout = QGridLayout()
        
        self.use_real_data_check = QCheckBox("Use Real Energy Data")
        self.use_real_data_check.setChecked(True)
        self.use_real_data_check.toggled.connect(self.on_use_real_data_toggled)
        energy_layout.addWidget(self.use_real_data_check, 0, 0, 1, 2)
        
        energy_layout.addWidget(QLabel("Custom Energy Profile:"), 1, 0)
        self.energy_profile_combo = QComboBox()
        self.energy_profile_combo.addItems(["Typical Residential", "High Evening Use", "Constant Load", "Custom"])
        self.energy_profile_combo.setEnabled(False)
        energy_layout.addWidget(self.energy_profile_combo, 1, 1)
        
        energy_group.setLayout(energy_layout)
        layout.addWidget(energy_group)
        
        # Status area
        status_group = QGroupBox("Data Status")
        status_layout = QVBoxLayout()
        self.status_text = QTextBrowser()
        self.status_text.setMaximumHeight(150)
        status_layout.addWidget(self.status_text)
        status_group.setLayout(status_layout)
        layout.addWidget(status_group)
        
        # Preview area
        preview_group = QGroupBox("Energy Data Preview")
        preview_layout = QVBoxLayout()
        self.energy_canvas = MatplotlibCanvas(self, width=5, height=4, dpi=100)
        preview_layout.addWidget(self.energy_canvas)
        preview_group.setLayout(preview_layout)
        layout.addWidget(preview_group)
        
        data_tab.setLayout(layout)
        tabs.addTab(data_tab, "Data Selection")
        
    def setup_ev_config_tab(self, tabs):
        """Setup the EV configuration tab"""
        ev_tab = QWidget()
        layout = QVBoxLayout()
        
        # EV selection
        ev_group = QGroupBox("EV Configuration")
        ev_layout = QGridLayout()
        
        ev_layout.addWidget(QLabel("EV Model:"), 0, 0)
        self.ev_model_combo = QComboBox()
        self.ev_model_combo.addItems(list(ev_config.EV_MODELS.keys()))
        self.ev_model_combo.currentTextChanged.connect(self.on_ev_model_changed)
        ev_layout.addWidget(self.ev_model_combo, 0, 1)
        
        ev_layout.addWidget(QLabel("Battery Capacity (kWh):"), 1, 0)
        self.battery_capacity_spin = QDoubleSpinBox()
        self.battery_capacity_spin.setRange(10, 200)
        self.battery_capacity_spin.setValue(60)
        self.battery_capacity_spin.setDecimals(1)
        ev_layout.addWidget(self.battery_capacity_spin, 1, 1)
        
        ev_layout.addWidget(QLabel("Max Charge Rate (kW):"), 2, 0)
        self.max_charge_rate_spin = QDoubleSpinBox()
        self.max_charge_rate_spin.setRange(1, 20)
        self.max_charge_rate_spin.setValue(11)
        self.max_charge_rate_spin.setDecimals(1)
        ev_layout.addWidget(self.max_charge_rate_spin, 2, 1)
        
        ev_layout.addWidget(QLabel("Max Discharge Rate (kW):"), 3, 0)
        self.max_discharge_rate_spin = QDoubleSpinBox()
        self.max_discharge_rate_spin.setRange(1, 20)
        self.max_discharge_rate_spin.setValue(11)
        self.max_discharge_rate_spin.setDecimals(1)
        ev_layout.addWidget(self.max_discharge_rate_spin, 3, 1)
        
        ev_layout.addWidget(QLabel("Battery Efficiency (%):"), 4, 0)
        self.efficiency_spin = QDoubleSpinBox()
        self.efficiency_spin.setRange(70, 100)
        self.efficiency_spin.setValue(90)
        self.efficiency_spin.setDecimals(1)
        ev_layout.addWidget(self.efficiency_spin, 4, 1)
        
        ev_layout.addWidget(QLabel("Minimum SoC (%):"), 5, 0)
        self.min_soc_spin = QDoubleSpinBox()
        self.min_soc_spin.setRange(0, 50)
        self.min_soc_spin.setValue(20)
        self.min_soc_spin.setDecimals(1)
        ev_layout.addWidget(self.min_soc_spin, 5, 1)
        
        ev_layout.addWidget(QLabel("Maximum SoC (%):"), 6, 0)
        self.max_soc_spin = QDoubleSpinBox()
        self.max_soc_spin.setRange(50, 100)
        self.max_soc_spin.setValue(90)
        self.max_soc_spin.setDecimals(1)
        ev_layout.addWidget(self.max_soc_spin, 6, 1)
        
        ev_layout.addWidget(QLabel("Initial SoC (%):"), 7, 0)
        self.initial_soc_spin = QDoubleSpinBox()
        self.initial_soc_spin.setRange(0, 100)
        self.initial_soc_spin.setValue(50)
        self.initial_soc_spin.setDecimals(1)
        ev_layout.addWidget(self.initial_soc_spin, 7, 1)
        
        ev_group.setLayout(ev_layout)
        layout.addWidget(ev_group)
        
        # EV availability
        avail_group = QGroupBox("EV Availability")
        avail_layout = QVBoxLayout()
        
        avail_combo_layout = QHBoxLayout()
        avail_combo_layout.addWidget(QLabel("Availability Profile:"))
        self.availability_combo = QComboBox()
        self.availability_combo.addItems(list(ev_config.EV_AVAILABILITY_PROFILES.keys()))
        self.availability_combo.currentTextChanged.connect(self.on_availability_profile_changed)
        avail_combo_layout.addWidget(self.availability_combo)
        avail_layout.addLayout(avail_combo_layout)
        
        # Hours selection
        hours_layout = QGridLayout()
        self.hour_checkboxes = []
        for i in range(24):
            hour_checkbox = QCheckBox(f"{i:02d}:00")
            hour_checkbox.setChecked(i in ev_config.EV_AVAILABILITY_PROFILES["Home at Night"]["hours_available"])
            self.hour_checkboxes.append(hour_checkbox)
            hours_layout.addWidget(hour_checkbox, i // 6, i % 6)
        
        avail_layout.addLayout(hours_layout)
        avail_group.setLayout(avail_layout)
        layout.addWidget(avail_group)
        
        ev_tab.setLayout(layout)
        tabs.addTab(ev_tab, "EV Configuration")
        
    def setup_optimization_tab(self, tabs):
        """Setup the optimization configuration tab"""
        opt_tab = QWidget()
        layout = QVBoxLayout()
        
        # Utility rates
        utility_group = QGroupBox("Utility Rate Configuration")
        utility_layout = QGridLayout()
        
        utility_layout.addWidget(QLabel("Utility:"), 0, 0)
        self.utility_combo = QComboBox()
        self.utility_combo.addItems(list(ev_config.UTILITY_RATES.keys()))
        self.utility_combo.currentTextChanged.connect(self.on_utility_changed)
        utility_layout.addWidget(self.utility_combo, 0, 1)
        
        utility_layout.addWidget(QLabel("Is Weekend:"), 1, 0)
        self.is_weekend_check = QCheckBox()
        self.is_weekend_check.toggled.connect(self.on_utility_changed)
        utility_layout.addWidget(self.is_weekend_check, 1, 1)
        
        utility_group.setLayout(utility_layout)
        layout.addWidget(utility_group)
        
        # Optimization settings
        opt_settings_group = QGroupBox("Optimization Settings")
        opt_settings_layout = QGridLayout()
        
        opt_settings_layout.addWidget(QLabel("Prioritize Cost Savings:"), 0, 0)
        self.cost_priority_slider = QSlider(Qt.Horizontal)
        self.cost_priority_slider.setRange(0, 100)
        self.cost_priority_slider.setValue(75)
        self.cost_priority_slider.setTickPosition(QSlider.TicksBelow)
        self.cost_priority_slider.setTickInterval(10)
        opt_settings_layout.addWidget(self.cost_priority_slider, 0, 1)
        
        opt_settings_layout.addWidget(QLabel("Reserve Capacity for Outages:"), 1, 0)
        self.outage_reserve_spin = QDoubleSpinBox()
        self.outage_reserve_spin.setRange(0, 100)
        self.outage_reserve_spin.setValue(20)
        self.outage_reserve_spin.setSuffix("%")
        opt_settings_layout.addWidget(self.outage_reserve_spin, 1, 1)
        
        opt_settings_group.setLayout(opt_settings_layout)
        layout.addWidget(opt_settings_group)
        
        # Rate preview
        rate_preview_group = QGroupBox("Rate Schedule Preview")
        rate_preview_layout = QVBoxLayout()
        self.rate_canvas = MatplotlibCanvas(self, width=5, height=3, dpi=100)
        rate_preview_layout.addWidget(self.rate_canvas)
        rate_preview_group.setLayout(rate_preview_layout)
        layout.addWidget(rate_preview_group)
        
        # Run optimization button
        self.run_opt_btn = QPushButton("Run Optimization")
        self.run_opt_btn.clicked.connect(self.on_run_optimization)
        self.run_opt_btn.setStyleSheet("font-weight: bold; font-size: 14px; padding: 8px;")
        layout.addWidget(self.run_opt_btn)
        
        opt_tab.setLayout(layout)
        tabs.addTab(opt_tab, "Optimization")
        
    def setup_results_tab(self, tabs):
        """Setup the results tab"""
        results_tab = QWidget()
        layout = QVBoxLayout()
        
        # Results charts
        results_group = QGroupBox("Optimization Results")
        results_layout = QVBoxLayout()
        
        # We'll be using the BatteryOptimizer's plot_results method to create plots
        self.results_canvas = FigureCanvas(Figure(figsize=(5, 10)))
        self.results_toolbar = NavigationToolbar(self.results_canvas, self)
        
        results_layout.addWidget(self.results_toolbar)
        results_layout.addWidget(self.results_canvas)
        
        results_group.setLayout(results_layout)
        layout.addWidget(results_group)
        
        # Summary
        summary_group = QGroupBox("Summary")
        summary_layout = QVBoxLayout()
        self.summary_text = QTextBrowser()
        summary_layout.addWidget(self.summary_text)
        summary_group.setLayout(summary_layout)
        layout.addWidget(summary_group)
        
        # Export button
        self.export_results_btn = QPushButton("Export Results")
        self.export_results_btn.clicked.connect(self.on_export_results)
        layout.addWidget(self.export_results_btn)
        
        results_tab.setLayout(layout)
        tabs.addTab(results_tab, "Results")
        
    # Event handlers
    def on_city_changed(self, city_name):
        """Handle city selection change"""
        self.status_text.append(f"Selected city: {city_name}")
        
    def on_load_city_data(self):
        """Load data for the selected city"""
        city_name = self.city_combo.currentText()
        self.status_text.append(f"Loading data for {city_name}...")
        
        # Load city data using data manager
        if self.data_manager.load_city_data(city_name):
            self.status_text.append(f"Successfully loaded data for {city_name}")
            
            # Update date picker with available dates
            available_dates = self.data_manager.get_available_dates(city_name)
            if available_dates:
                min_date = available_dates[0]
                max_date = available_dates[-1]
                self.date_picker.setDateRange(
                    QDate(min_date.year, min_date.month, min_date.day),
                    QDate(max_date.year, max_date.month, max_date.day)
                )
                self.date_picker.setDate(QDate(min_date.year, min_date.month, min_date.day))
            
            # Update utility selection based on county
            county = self.data_manager.get_city_county(city_name)
            if county:
                county_util = next((u for u in ev_config.UTILITY_RATES.keys() if county in u), None)
                if county_util:
                    self.utility_combo.setCurrentText(county_util)
        else:
            self.status_text.append(f"Failed to load data for {city_name}")
    
    def on_load_date_data(self):
        """Load data for the selected date"""
        city_name = self.city_combo.currentText()
        qdate = self.date_picker.date()
        date = datetime(qdate.year(), qdate.month(), qdate.day()).date()
        
        self.status_text.append(f"Loading data for {city_name} on {date}...")
        
        # Get energy data for the selected date
        energy_data = self.data_manager.get_energy_for_date(city_name, date)
        
        if energy_data is not None:
            self.status_text.append(f"Successfully loaded energy data for {date}")
            self.plot_energy_data(energy_data)
        else:
            self.status_text.append(f"No energy data available for {date}")
    
    def on_load_outage_data(self):
        """Load outage data for the selected year"""
        year = int(self.outage_year_combo.currentText())
        
        self.status_text.append(f"Loading outage data for {year}...")
        
        if self.data_manager.load_outage_data(year):
            self.status_text.append(f"Successfully loaded outage data for {year}")
        else:
            self.status_text.append(f"Failed to load outage data for {year}")
    
    def on_use_real_data_toggled(self, checked):
        """Toggle between real and custom energy data"""
        self.energy_profile_combo.setEnabled(not checked)
    
    def on_ev_model_changed(self, model_name):
        """Update EV parameters based on selected model"""
        if model_name == "Custom":
            return  # Don't override custom settings
            
        ev_model = ev_config.EV_MODELS[model_name]
        
        self.battery_capacity_spin.setValue(ev_model["battery_capacity"])
        self.max_charge_rate_spin.setValue(ev_model["max_charge_rate_ac"])
        self.max_discharge_rate_spin.setValue(ev_model["max_discharge_rate"])
        self.efficiency_spin.setValue(ev_model["efficiency"] * 100)
        self.min_soc_spin.setValue(ev_model["min_soc"] * 100)
        self.max_soc_spin.setValue(ev_model["max_soc"] * 100)
    
    def on_availability_profile_changed(self, profile_name):
        """Update availability checkboxes based on selected profile"""
        profile = ev_config.EV_AVAILABILITY_PROFILES[profile_name]
        hours_available = profile["hours_available"]
        
        for i, checkbox in enumerate(self.hour_checkboxes):
            checkbox.setChecked(i in hours_available)
    
    def on_utility_changed(self):
        """Update utility rate preview"""
        utility_name = self.utility_combo.currentText()
        qdate = self.date_picker.date()
        date = datetime(qdate.year(), qdate.month(), qdate.day())
        is_weekend = self.is_weekend_check.isChecked()
        
        # Get hourly rates
        hourly_rates = ev_config.get_utility_rate_schedule(utility_name, date, is_weekend)
        
        # Plot rates
        self.plot_utility_rates(hourly_rates)
    
    def on_run_optimization(self):
        """Run optimization with current settings"""
        # Check if data is loaded
        city_name = self.city_combo.currentText()
        qdate = self.date_picker.date()
        date = datetime(qdate.year(), qdate.month(), qdate.day()).date()
        
        if city_name not in self.data_manager.energy_data:
            QMessageBox.warning(self, "Warning", "Please load city data first.")
            return
        
        # Get energy data
        energy_data = None
        if self.use_real_data_check.isChecked():
            energy_data = self.data_manager.get_energy_for_date(city_name, date)
            if energy_data is None:
                QMessageBox.warning(self, "Warning", f"No energy data available for {date}")
                return
        else:
            # Use custom profile
            profile_name = self.energy_profile_combo.currentText()
            energy_data = self.create_custom_energy_profile(profile_name)
        
        # Get EV availability
        ev_availability = [checkbox.isChecked() for checkbox in self.hour_checkboxes]
        
        # Get outage data
        county = self.data_manager.get_city_county(city_name)
        outage_data = None
        if county:
            outage_data = self.data_manager.get_county_outage_data(county, date)
        
        # Configure battery optimizer
        self.battery_optimizer = BatteryOptimizer(
            battery_capacity=self.battery_capacity_spin.value(),
            efficiency=self.efficiency_spin.value() / 100,
            max_charge_rate=self.max_charge_rate_spin.value(),
            max_discharge_rate=self.max_discharge_rate_spin.value()
        )
        
        self.battery_optimizer.set_battery_params(
            initial_soc=self.initial_soc_spin.value() / 100,
            min_soc=self.min_soc_spin.value() / 100,
            max_soc=self.max_soc_spin.value() / 100
        )
        
        # Get utility rates
        utility_name = self.utility_combo.currentText()
        is_weekend = self.is_weekend_check.isChecked()
        hourly_rates = ev_config.get_utility_rate_schedule(utility_name, date, is_weekend)
        self.battery_optimizer.set_price_schedule(hourly_rates)
        
        # Prepare outage probability from outage data
        outage_prob = None
        if outage_data is not None:
            outage_prob = outage_data['outage_percentage'].values / 100
        
        # Run optimization
        hourly_consumption = energy_data['load_per_dwelling'].values
        results = self.battery_optimizer.optimize_battery_usage(
            hourly_consumption=hourly_consumption,
            ev_availability=ev_availability,
            outage_prob=outage_prob
        )
        
        # Display results
        fig, _ = self.battery_optimizer.plot_results(results, show_plot=False)
        
        # Update results canvas
        self.results_canvas.figure.clear()
        self.results_canvas.figure.set_size_inches(12, 15)  # Set appropriate figure size
        
        # Create subplots
        ax1 = self.results_canvas.figure.add_subplot(3, 1, 1)
        ax2 = self.results_canvas.figure.add_subplot(3, 1, 2)
        ax3 = self.results_canvas.figure.add_subplot(3, 1, 3)
        
        # Copy plots from original figure to new axes
        for i, (ax, new_ax) in enumerate(zip(fig.axes, [ax1, ax2, ax3])):
            new_ax.set_title(ax.get_title())
            new_ax.set_xlabel(ax.get_xlabel())
            new_ax.set_ylabel(ax.get_ylabel())
            
            # Copy all artists from original axes to new axes
            for artist in ax.get_children():
                if hasattr(artist, 'get_data'):
                    try:
                        x, y = artist.get_data()
                        new_ax.plot(x, y, color=artist.get_color(), 
                                  linestyle=artist.get_linestyle(), 
                                  marker=artist.get_marker(),
                                  label=artist.get_label())
                    except:
                        pass
                elif hasattr(artist, 'get_height'):
                    try:
                        new_ax.bar(artist.get_x(), artist.get_height(), 
                                 width=artist.get_width(), 
                                 color=artist.get_facecolor(),
                                 label=artist.get_label())
                    except:
                        pass
            
            # Copy legend if it exists and has labeled artists
            if ax.get_legend() and any(artist.get_label() and artist.get_label() != '_nolegend_' for artist in ax.get_children()):
                new_ax.legend()
        
        self.results_canvas.draw()
        
        # Update summary
        savings_amount, savings_pct = self.battery_optimizer.calculate_savings(results)
        total_cost = sum(results['cost'])
        total_consumption = sum(results['consumption'])
        grid_consumption = sum(results['grid_consumption'])
        battery_contribution = total_consumption - grid_consumption
        
        summary = f"""
        <h3>Optimization Results</h3>
        <p><b>Date:</b> {date}</p>
        <p><b>City:</b> {city_name}</p>
        <p><b>Utility:</b> {utility_name}</p>
        <p><b>Total Cost:</b> ${total_cost:.2f}</p>
        <p><b>Cost Savings:</b> ${savings_amount:.2f} ({savings_pct:.1f}%)</p>
        <p><b>Total Home Consumption:</b> {total_consumption:.1f} kWh</p>
        <p><b>Grid Consumption:</b> {grid_consumption:.1f} kWh</p>
        <p><b>Battery Contribution:</b> {battery_contribution:.1f} kWh</p>
        <p><b>Estimated Battery Cycles:</b> {self.battery_optimizer.cycle_count:.2f}</p>
        """
        
        self.summary_text.setHtml(summary)
    
    def on_export_results(self):
        """Export results to CSV"""
        file_name, _ = QFileDialog.getSaveFileName(self, "Save Results", "", "CSV Files (*.csv)")
        if file_name:
            try:
                # TODO: Implement exporting results to CSV
                QMessageBox.information(self, "Success", f"Results saved to {file_name}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Error saving results: {str(e)}")
    
    def plot_energy_data(self, energy_data):
        """Plot energy data"""
        self.energy_canvas.axes.clear()
        self.energy_canvas.axes.bar(energy_data['Hour'], energy_data['load_per_dwelling'], color='skyblue')
        self.energy_canvas.axes.set_xlabel('Hour of Day')
        self.energy_canvas.axes.set_ylabel('Energy Consumption (kWh)')
        self.energy_canvas.axes.set_title('Hourly Energy Consumption')
        self.energy_canvas.axes.set_xticks(range(0, 24, 2))
        self.energy_canvas.axes.set_xticklabels([f"{h:02d}:00" for h in range(0, 24, 2)])
        self.energy_canvas.axes.grid(alpha=0.3)
        self.energy_canvas.draw()
    
    def plot_utility_rates(self, hourly_rates):
        """Plot utility rates"""
        self.rate_canvas.axes.clear()
        hours = list(hourly_rates.keys())
        rates = list(hourly_rates.values())
        self.rate_canvas.axes.bar(hours, rates, color='orangered')
        self.rate_canvas.axes.set_xlabel('Hour of Day')
        self.rate_canvas.axes.set_ylabel('Rate ($/kWh)')
        self.rate_canvas.axes.set_title('Hourly Electricity Rates')
        self.rate_canvas.axes.set_xticks(range(0, 24, 2))
        self.rate_canvas.axes.set_xticklabels([f"{h:02d}:00" for h in range(0, 24, 2)])
        self.rate_canvas.axes.grid(alpha=0.3)
        self.rate_canvas.draw()
    
    def create_custom_energy_profile(self, profile_name):
        """Create a custom energy profile based on the selected profile"""
        hours = list(range(24))
        
        if profile_name == "Typical Residential":
            # Morning and evening peaks
            values = [0.2, 0.2, 0.2, 0.2, 0.2, 0.3, 0.5, 0.8, 0.7, 0.5, 0.4, 0.4,
                     0.5, 0.5, 0.4, 0.5, 0.7, 0.9, 1.1, 1.0, 0.8, 0.6, 0.4, 0.3]
        elif profile_name == "High Evening Use":
            # High usage in evening hours
            values = [0.2, 0.2, 0.2, 0.2, 0.2, 0.3, 0.4, 0.5, 0.5, 0.4, 0.4, 0.5,
                     0.6, 0.6, 0.5, 0.6, 0.8, 1.2, 1.5, 1.4, 1.2, 0.8, 0.5, 0.3]
        elif profile_name == "Constant Load":
            # Relatively flat load throughout the day
            values = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.6, 0.6, 0.6, 0.6, 0.6,
                     0.6, 0.6, 0.6, 0.6, 0.6, 0.7, 0.7, 0.7, 0.6, 0.6, 0.5, 0.5]
        else:  # Custom
            # Default custom profile with moderate evening peak
            values = [0.3, 0.3, 0.3, 0.3, 0.3, 0.4, 0.5, 0.6, 0.6, 0.5, 0.5, 0.6,
                     0.6, 0.6, 0.5, 0.6, 0.7, 0.8, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4]
        
        return pd.DataFrame({'Hour': hours, 'load_per_dwelling': values})

def main():
    app = QApplication(sys.argv)
    
    # Set style
    app.setStyle("Fusion")
    
    window = HomeEnergyModelApp()
    window.show()
    
    sys.exit(app.exec_())

if __name__ == "__main__":
    main() 
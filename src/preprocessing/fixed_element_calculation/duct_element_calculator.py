import numpy as np
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# logging.disable(logging.INFO)
# logging.disable(logging.WARNING)


class DuctCalc:
    """Stateful calculator for computing the pressure loss factor and acoustics of a rectangular duct within a room (incl. optional branch)."""

    # ------------------------- lifecycle -------------------------
    def __init__(
        self,
        width: float,
        height: float,
        volume_flow: float,
        ignore_components,
        include_acoustics,
    ):
        self.width = float(width)
        self.height = float(height)
        self.rho = 1.2
        self.volume_flow_ratio = 1.0  # running multiplier along the path
        self.zeta = 0.0  # zeta referred to volume flow, not velocity. also the density is already included. it is used for the equation: dp = zeta * q^2
        self.history = []  # per-element snapshots
        self.dampening = np.array([0.0] * 8)
        self.flow_noise = np.array([-10.0] * 8)
        self.volume_flow = volume_flow
        self.f_m = np.array([63, 125, 250, 500, 1000, 2000, 4000, 8000])
        self.ignore_components = ignore_components
        self.include_acoustics = include_acoustics

    def reset(self, *, width: float = None, height: float = None) -> None:
        """Reset running state (and optionally width/height)."""
        if width is not None:
            self.width = float(width)
        if height is not None:
            self.height = float(height)
        self.volume_flow_ratio = 1.0
        self.zeta = 0.0
        self.history = []

    # ------------------------- helpers --------------------------
    def _apply_overrides(self, element: dict) -> None:
        """Update current width/height if element provides overrides."""
        if "width" in element:
            self.width = element["width"]
        if "height" in element:
            self.height = element["height"]

    def _dispatch(self):
        """Mapping from element name -> bound method."""
        return {
            "duct": self.duct,
            "fire_damper": self.fire_damper,
            "louver_damper": self.louver_damper,
            "rect_bending": self.rectangular_bending,
            "round_bending": self.round_bending,
            "volume_regulator": self.volume_regulator,
            "volume_flow_controller": self.volume_regulator,
            # "duct_silencer": self.duct_silencer,
            "rect_branch": self.rectangular_branch,
            "air_diffuser": self.air_diffuser,
            "reheater": self.reheater,
            # "splitter_silencer": self.splitter_silencer,
        }

    def _calculate_zeta_factor(self, *dimensions: float) -> float:
        if len(dimensions) == 2:
            w, h = dimensions
            return 1 / (w * h) ** 2 * self.volume_flow_ratio**2 * self.rho / 2
        elif len(dimensions) == 1:
            diameter = dimensions[0]
            return 1 / (diameter**2/4*np.pi) ** 2 * self.volume_flow_ratio**2 * self.rho / 2
        else:
            raise ValueError(f"Input dimension is supposed to be one or two, was {len(dimensions)}")
        

    # ------------------------- components -----------------------
    def zeta_value(self, element: dict) -> None:
        """transform dimensionless zeta relative to velocity^2 --> zeta relative to volume_flow^2 (m³/s)^2"""
        self._apply_overrides(element)
        self.zeta += element["value"] * self._calculate_zeta_factor(
            self.width, self.height
        )

    def level_add(self, oktav_spl):
        """
        Pegel Addition beliebiger Element einer Liste oder eines 1D arrays
        """
        return 10 * np.log10(np.sum([10 ** (0.1 * (x)) for x in oktav_spl]))

    def series_add_dampening(self, D):
        return sum(D)

    def series_add_flow_noise(self, D, Lf):
        return self.level_add(
            [Lf[i] - sum([D[j] for j in range(i + 1, len(Lf))]) for i in range(len(Lf))]
        )

    def series_add_dampening_list(self, list_of_dampenings):
        return [self.series_add_dampening(group) for group in zip(*list_of_dampenings)]

    def series_add_flow_noise_list(self, list_of_dampenings, list_of_flow_noises):
        return [
            self.series_add_flow_noise(group1, group2)
            for group1, group2 in zip(
                zip(*list_of_dampenings), zip(*list_of_flow_noises)
            )
        ]

    def duct(self, element: dict) -> None:
        """VDI 3803 Bl.6 - straight duct round or rect."""
        self._apply_overrides(element)
        lamb = 0.02  #friction factor
        L = element["length"]
        if "diameter" in element:
            d_h = element["diameter"]
            self.zeta += (lamb * L / d_h) * self._calculate_zeta_factor(d_h)
        else:
            w, h = self.width, self.height
            d_h = 2 * h * w / (h + w)  # hydraulic diameter
            self.zeta += (lamb * L / d_h) * self._calculate_zeta_factor(w, h)
        
        if self.include_acoustics:
            dampening, flow_noise = self.calculate_rectangular_duct_acoustics(element)
            #logging.info(dampening)
            self.dampening = self.series_add_dampening_list([self.dampening, dampening])
            self.flow_noise = self.series_add_flow_noise_list(
                [self.dampening, dampening], [self.flow_noise, flow_noise]
            )

    def calculate_rectangular_duct_acoustics(self, element):
        """
        Berechnung von rechteckigen Luftleitungen VDI 2081 Blatt 1 :2022-04
        Schallleistung analog zu runder Luftleitung 10.3.1
        Gleichung (24), (25)
        Dämpfung mit längster Seitenlänge 11.1
        Tabelle 13
        """
        length = element["length"]

        if "diameter" in element:
            
            diameter = element["diameter"]
            area = np.pi*diameter**2/4
            ## dampening:

            if (0.1 <= diameter) and (diameter <= 0.2):
                dampening = np.array([0.1, 0.1, 0.15, 0.15, 0.3, 0.3, 0.3, 0.3]) * length
            elif (0.2 < diameter) and (diameter <= 0.4):
                dampening = np.array([0.05, 0.1, 0.1, 0.15, 0.2, 0.2, 0.2, 0.2]) * length
            elif (0.4 < diameter) and (diameter <= 0.8):
                dampening = np.array([0, 0.05, 0.05, 0.1, 0.15, 0.15, 0.15, 0.15]) * length
            elif (0.8 < diameter) and (diameter <= 1):
                dampening = (
                    np.array([0, 0, 0, 0.05, 0.05, 0.05, 0.05, 0.05]) * length
                )
            else:
                dampening = np.zeros(8)

        else:
            area = self.width * self.height

            longest_side = np.max([self.width, self.height])

            ## dampening:

            if (0.1 <= longest_side) and (longest_side <= 0.2):
                dampening = np.array([0.6, 0.6, 0.45, 0.3, 0.3, 0.3, 0.3, 0.3]) * length
            elif (0.2 < longest_side) and (longest_side <= 0.4):
                dampening = np.array([0.6, 0.6, 0.45, 0.3, 0.2, 0.2, 0.2, 0.2]) * length
            elif (0.4 < longest_side) and (longest_side <= 0.8):
                dampening = np.array([0.6, 0.6, 0.3, 0.15, 0.15, 0.15, 0.15, 0.15]) * length
            elif (0.8 < longest_side) and (longest_side <= 1):
                dampening = (
                    np.array([0.45, 0.3, 0.15, 0.1, 0.05, 0.05, 0.05, 0.05]) * length
                )
            else:
                dampening = np.zeros(8)

        
        velocity = self.volume_flow / area

        

        Lw = 16.5 + 48.2 * np.log10(velocity) + 10 * np.log10(area)

        Delta_L_W_Okt = -6.24 - 21.75 * np.log10(0.228 + 0.094 * self.f_m / velocity)
        Delta_L_W_Okt_Gesamt = self.level_add(Delta_L_W_Okt)

        flow_noise = Lw + Delta_L_W_Okt - Delta_L_W_Okt_Gesamt

        return dampening, flow_noise

    def rectangular_bending(self, element: dict) -> None:
        """VDI 3803 Bl.6 - Typ A02 (placeholder fit)."""
        self._apply_overrides(element)
        w, h = self.width, self.height
        alpha = element["bending_angle"]
        Rb = element["bending_radius"]
        n = element["n_bendings"]
        A = lambda a: 1.6094 - 1.60868 * np.exp(-0.01089 * a)
        B = lambda R: 0.21 / R**2.5
        C = lambda ab: (
            -1.03663e-4 * ab**5
            + 0.00338 * ab**4
            - 0.04277 * ab**3
            + 0.25496 * ab**2
            - 0.66296 * ab
            + 1.4499
        )
        self.zeta += (
            A(alpha) * B(Rb / w) * C(h / w) * n * self._calculate_zeta_factor(w, h)
        )

        
        hydraulic_diameter = 4 * w*h / (2 * (w + h))

        if self.include_acoustics:
            dampening, flow_noise = self.calculate_bending_acoustics(
                element, hydraulic_diameter
            )
            #logging.info(dampening)
            self.dampening = self.series_add_dampening_list([self.dampening, dampening])
            self.flow_noise = self.series_add_flow_noise_list(
                [self.dampening, dampening], [self.flow_noise, flow_noise]
            )


    def round_bending(self, element: dict) -> None:
        """VDI 3803 Bl.6 - Typ A01."""
        self._apply_overrides(element)
        diameter = element["diameter"]
        alpha = element["bending_angle"]
        Rb = element["bending_radius"]
        n = element["n_bendings"]
        A = lambda a: 1.6094 - 1.60868 * np.exp(-0.01089 * a)
        B = lambda R: 0.21 / R**2.5
        C = 1
        self.zeta += (
            A(alpha) * B(Rb / diameter) * C * n * self._calculate_zeta_factor(diameter)
        )
        if self.include_acoustics:
            dampening, flow_noise = self.calculate_bending_acoustics(
                element, diameter
            )
            #logging.info(dampening)
            self.dampening = self.series_add_dampening_list([self.dampening, dampening])
            self.flow_noise = self.series_add_flow_noise_list(
                [self.dampening, dampening], [self.flow_noise, flow_noise]
            )

    def calculate_bending_acoustics(self, element, hydraulic_diameter):
        """
        Berechnung von runden / rechteckigen Umlenkungen VDI 2081 Blatt 1 :2022-04
        Schallleistung 10.3.2.1
        Gleichung (28), (29); Faktor K wird vernachlässigt (30)
        Berechnung der Strouhal-Zahl mit hydraulischen Durchmesser, Ersatzdurchmesser in (28) gemäß 10.3.2.2
        Dämpfung 11.2
        Gleichung (40)
        Tabelle 14
        """
        area = self.width * self.height
        diameter = np.sqrt(4 / np.pi * area)
        velocity = self.volume_flow / area
        bending_radius = element["bending_radius"]
        n_bendings = element["n_bendings"]
        longest_side = max(self.width, self.height)

        delta_f = np.array([45, 88, 177, 254, 707, 1414, 2828, 5657])
        f_o = np.array([89, 177, 354, 707, 1414, 2828, 5657, 11314])

        Strouhal_number = self.f_m * hydraulic_diameter / velocity

        if any(Strouhal_number < 1):
            Strouhal_number = np.where(Strouhal_number < 1, 1, Strouhal_number)
            
        # K = 13.9*(3.43-np.log10(Strouhal_number))*(0.15-bending_radius/diameter)
        
        flow_noise_star = 12 - 21.5 * (np.log10(Strouhal_number)) ** 1.268

        flow_noise = (
            flow_noise_star
            + 10 * np.log10(delta_f)
            + 30 * np.log10(diameter)
            + 50 * np.log10(velocity)
        )  # + K

        ### dampening:
        ## section 11.2 a)
        speed_of_sound = 340

        f_G = speed_of_sound / (2 * longest_side)

        def find_category(value, intervals):
            """
            Finds the category where the value fits in
            """
            index = np.searchsorted(intervals, value, side="right")
            return index

        ## section 11.2 b)
        category = find_category(f_G, f_o)

        ## section 11.2 c)
        def custom_shift(arr, shift_val):
            # Create an array of zeros with the same shape as arr
            shifted = np.zeros_like(arr)

            # If shifting to the right
            if shift_val > 0:
                shifted[shift_val:] = arr[:-shift_val]
            # If shifting to the left
            elif shift_val < 0:
                shifted[:shift_val] = arr[-shift_val:]
            # If no shift, simply return the original array
            else:
                return arr

            return shifted

        dampening_levels = np.array([0, 1, 2, 3, 3, 3, 3, 3, 3])
        dampening = custom_shift(dampening_levels, category - 2)[:-1]

        dampening = self.series_add_dampening_list([dampening] * n_bendings)

        flow_noise = self.series_add_flow_noise_list(
            [dampening] * n_bendings,
            [flow_noise] * n_bendings,
        )

        return dampening, flow_noise

    

    def air_diffuser(self, element):
        self._apply_overrides(element)
        self.zeta += element["zeta"] * self._calculate_zeta_factor(
            self.width, self.height
        )

        if self.include_acoustics:
            dampening, flow_noise = self.calculate_air_diffuser_acoustics(element)
            #logging.info(dampening)
            self.dampening = self.series_add_dampening_list([self.dampening, dampening])
            self.flow_noise = self.series_add_flow_noise_list(
                [self.dampening, dampening], [self.flow_noise, flow_noise]
            )

    def reheater(self, element):
        self._apply_overrides(element)
        self.zeta += element["zeta"] * self._calculate_zeta_factor(
            self.width, self.height
        )

        if self.include_acoustics:
            dampening, flow_noise = np.array([0] * 8), np.array([-10] * 8)
            #logging.info(dampening)
            self.dampening = self.series_add_dampening_list([self.dampening, dampening])
            self.flow_noise = self.series_add_flow_noise_list(
                [self.dampening, dampening], [self.flow_noise, flow_noise]
            )

    def calculate_air_diffuser_acoustics(self, element):
        """
        Berechnung von Luftdurchlässen VDI 2081 Blatt 1 :2022-04
        Schallleistung 10.3.3
        Gleichung (31), (32)
        Bestimmung Zeta-Wert Tabelle 11
        Dämpfung 11.6
        Gleichung (43)
        Bild 16
        """

        zeta = element["zeta"]

        duct_area = self.width * self.height
        diffuser_type = element["diffuser_type"]

        outlet_location = element["outlet_location"]

        
        # es gibt noch andere diffuser, aber die wurden nicht benötigt!
        if diffuser_type == "Textilauslass":
            area = element["length"] * element["diameter"] * np.pi
            velocity = self.volume_flow / area
            total_pressure_difference = velocity * 2000
        elif diffuser_type in (
            "Lamellendurchlass ungedrosselt",
            "quadratischer Vierwegedurchlass",
            "Gitterstahlprofil ungedrosselt",
            "Dralldurchlass 45°"
        ):
            if diffuser_type == "quadratischer Vierwegedurchlass":
                if "width" in element:
                    area = element["width"] * element["height"]
                else:
                    # if no width or height is given then the last used width or height is taken
                    area = self.width * self.height
            elif diffuser_type in ["Gitterstahlprofil ungedrosselt", "Dralldurchlass 45°"]:
                area = element["outlet_area"]
            else:
                area = element["diameter"] ** 2 / 4 * np.pi
            velocity = self.volume_flow / area
            total_pressure_difference = zeta * self.rho / 2 * velocity**2
        else:
            raise ValueError(
                f"diffuser type should be one of the two, not {diffuser_type}"
            )

        # total_pressure_difference = zeta * air_density / 2 * velocity**2
        L_w = (
            2.7 * np.log10(self.volume_flow * 3600)
            + 27.9 * np.log10(total_pressure_difference)
            - 5.4
        )

        Delta_L_w_Okt = -(
            71.72 - 67.37 / (1 + (self.f_m / velocity / zeta / 363.74) ** 1.1)
        )

        Delta_L_W_Okt_Gesamt = self.level_add(Delta_L_w_Okt)
        flow_noise = L_w + Delta_L_w_Okt - Delta_L_W_Okt_Gesamt

        ##dampening

        if outlet_location == "im Raum":
            solid_angle = 4 * np.pi
        elif outlet_location == "in der Fläche":
            solid_angle = 2 * np.pi
        elif outlet_location == "an Kante":
            solid_angle = np.pi
        elif outlet_location == "in Ecke":
            solid_angle = 0, 5 * np.pi
        else:
            raise Exception("Define outlet_location")

        speed_of_sound = 340

        delta_Lw = 10 * np.log10(
            1 + (speed_of_sound / (4 * np.pi * self.f_m)) ** 2 * solid_angle / duct_area
        )
        dampening = np.where(delta_Lw < 15, delta_Lw, 15)

        return dampening, flow_noise

    def louver_damper(self, element: dict) -> None:
        """Jalousieklappen, A33 gleichläufig."""
        self._apply_overrides(element)
        w, h = self.width, self.height
        K1, K2, K3 = 0.500, 0.120, 0.085
        alpha = element["angle"] if "angle" in element else 0
        self.zeta += (K1 + K2 * np.exp(K3 * alpha)) * self._calculate_zeta_factor(w, h)
        if self.include_acoustics:
            dampening, flow_noise = self.calculate_louver_damper_acoustics(element)
            #logging.info(dampening)
            self.dampening = self.series_add_dampening_list([self.dampening, dampening])
            self.flow_noise = self.series_add_flow_noise_list(
                [self.dampening, dampening], [self.flow_noise, flow_noise]
            )

    def calculate_louver_damper_acoustics(self, element):
        """
        VDI 2081 | 1 eq. 36
        """
        area = self.width * self.height

        velocity = self.volume_flow / area

        flow_noise = (
            10 + 60 * np.log10(velocity) + 28 * np.log10(0.1 + 1) + 10 * np.log10(area)
        )

        # expecting lamellenhoehe 130 - 200 mm
        K = [23.2, -46.08, 41.82, -17.03, 3.26, -0.182]

        Z = self.f_m / velocity / 0.1**0.3

        dampening = np.zeros(8)

        delta_LW_Okt = -np.sum([K[i] * np.log10(Z) ** i for i in range(6)], axis=0)

        return dampening, flow_noise + delta_LW_Okt - self.level_add(delta_LW_Okt)

    def fire_damper(self, element: dict) -> None:
        """Drossel-und Stellklappe mit einer einzelnen Klappe. Nicht nur für fire damper gültig. A32."""
        self._apply_overrides(element)
        w, h = self.width, self.height
        K1, K2, K3 = 0.200, 0.100, 0.115
        alpha = element["angle"]
        self.zeta += (K1 + K2 * np.exp(K3 * alpha)) * self._calculate_zeta_factor(w, h)
        if self.include_acoustics:
            dampening, flow_noise = self.calculate_louver_damper_acoustics(element)
            #logging.info(dampening)
            self.dampening = self.series_add_dampening_list([self.dampening, dampening])
            self.flow_noise = self.series_add_flow_noise_list(
                [self.dampening, dampening], [self.flow_noise, flow_noise]
            )

    def volume_regulator(self, element: dict) -> None:
        """A32-like placeholder; does not change volume_flow_ratio here."""
        self._apply_overrides(element)
        w, h = self.width, self.height
        K1, K2, K3 = 0.200, 0.100, 0.115
        alpha = element["angle"]
        self.zeta += (K1 + K2 * np.exp(K3 * alpha)) * self._calculate_zeta_factor(w, h)

    def rectangular_branch(self, element: dict) -> None:
        """A24 - T-Stück; updates volume_flow_ratio."""

        direction = element["direction"]
        if direction == "straight":
            K1, K2, K3 = 183.3, 0.06, 0.17
        elif direction == "bend":
            K1, K2, K3 = 301.95, 0.06, 0.75
        else:
            raise ValueError(f"Unknown branch direction: {direction!r}")
        # update running volume-flow ratio
        

        if "main_width" in element:
            A_main = element["main_width"] * element["main_height"]
            width, height = element["target_width"], element["target_height"]
            A_branch = width*height
            vfr = element["target_volume_flow"] / element["main_volume_flow"] if "target_volume_flow" in element else element["volume_flow_ratio"]
            
        else:
            A_main = self.width * self.height
            width, height = element["width"], element["height"]
            A_branch = width*height
            vfr = element["volume_flow_ratio"]
            
        self.volume_flow_ratio *= vfr
        self.volume_flow *= vfr
        w_ratio = vfr * A_main / A_branch
        zeta_local = K1 * np.exp(-w_ratio / K2) + K3
        self.zeta += zeta_local * self._calculate_zeta_factor(
            width, height
        )
        if self.include_acoustics:
            dampening, flow_noise = self.calculate_branch_acoustics(element)
            #logging.info(dampening)
            self.dampening = self.series_add_dampening_list([self.dampening, dampening])
            self.flow_noise = self.series_add_flow_noise_list(
                [self.dampening, dampening], [self.flow_noise, flow_noise]
            )

        self._apply_overrides(element)

    def round_branch(self, element: dict) -> None:
        """A22 - T-Stück; updates volume_flow_ratio."""

        if "main_width" in element:
            A_main = element["main_diameter"]**2/4*np.pi
            diameter = element["target_diameter"]
            A_branch = diameter**2/4*np.pi
            vfr = element["target_volume_flow"] / element["main_volume_flow"]
            
        else:
            A_main = self.width * self.height
            diameter = element["diameter"]
            A_branch = diameter**2/4*np.pi
            vfr = element["volume_flow_ratio"]

        self.volume_flow_ratio *= vfr
        self.volume_flow *= vfr
        w_ratio = vfr * A_main / A_branch

        direction = element["direction"]
        if direction == "straight":
            KD = 0.4
            zeta_local = KD*(1-w_ratio)**2/(w_ratio)**2
        elif direction == "bend":
            Y0,A1,A2,A3,t1,t2,t3 = 0.24, 131.28, 0.08552, 29542.98, 0.01439, 15.98, 0.24519
            zeta_local = Y0 + A1*np.exp(-w_ratio/t1)+ A2 *np.exp(-w_ratio/t2) + A3*np.exp(-w_ratio/t3)
        else:
            raise ValueError(f"Unknown branch direction: {direction!r}")
        # update running volume-flow ratio
        
        self.zeta += zeta_local * self._calculate_zeta_factor(
            diameter
        )
        if self.include_acoustics:
            dampening, flow_noise = self.calculate_branch_acoustics(element)
            #logging.info(dampening)
            self.dampening = self.series_add_dampening_list([self.dampening, dampening])
            self.flow_noise = self.series_add_flow_noise_list(
                [self.dampening, dampening], [self.flow_noise, flow_noise]
            )
        
        self.volume_flow_ratio *= vfr
        self.volume_flow *= vfr
        self._apply_overrides(element)

    def calculate_branch_acoustics(
        self,
        element,
    ):
        """
        Berechnung von rechteckigen Verzweigungen VDI 2081 Blatt 1 :2022-04
        Schallleistung 10.3.2.1
        Gleichung (28), (29), (30)
        Berechnung der Strouhal-Zahl mit hydraulischen Durchmesser, Ersatzdurchmesser in (28) gemäß 10.3.2.2
        Dämpfung 11.4, 11.2
        Gleichung (42), (40)
        Tabelle 14
        Bei gleichzeitiger Umlenkung der betrachteten Strömung
        """

        if "main_width" in element:
            A_branch = element["target_width"] * element["target_height"]
            A_neighbour_branch = element["neighbour_width"] * element["neighbour_height"]
            A_main = element["main_width"] * element["main_height"]
            vfr = element["target_volume_flow"] / element["main_volume_flow"] if "target_volume_flow" in element else element["volume_flow_ratio"]
        else:
            A_branch = element["width"] * element["height"]
            A_neighbour_branch = A_branch
            A_main = self.width*self.height
            vfr = element["volume_flow_ratio"]

        velocity_main_line = self.volume_flow / A_main

        diameter_branch = np.sqrt(4 / np.pi * A_branch)
        velocity_branch = self.volume_flow * vfr / A_branch

        f_m = np.array([63, 125, 250, 500, 1000, 2000, 4000, 8000])
        f_o = np.array([89, 177, 354, 707, 1414, 2828, 5657, 11314])

        if velocity_branch == 0:
            logging.warning(
                "Velocity in branch is zero! Setting the flow noise to 0."
            )

            flow_noise = -10 * np.zeros(8)

        else:
            bending_radius = 0.15  # hardcoded

            # hydraulischer Durchmesser ? 10.3.3.2
            hydraulic_diameter = np.sqrt(4 / np.pi * A_branch)

            delta_f = np.array([45, 88, 177, 254, 707, 1414, 2828, 5657])

            Strouhal_number = f_m * hydraulic_diameter / velocity_branch

            if any(Strouhal_number < 1):
                Strouhal_number = np.where(Strouhal_number < 1, 1, Strouhal_number)
            #     raise Exception("Strouhal_number must be larger than 1")

            K = (
                13.9
                * (3.43 - np.log10(Strouhal_number))
                * (0.15 - bending_radius / diameter_branch)
            )

            flow_noise_star = (
                12
                - 21.5 * (np.log10(Strouhal_number)) ** 1.268
                + (32 + 13 * np.log10(Strouhal_number))
                * np.log10(velocity_main_line / velocity_branch)
            )

            flow_noise = (
                flow_noise_star
                + 10 * np.log10(delta_f)
                + 30 * np.log10(diameter_branch)
                + 50 * np.log10(velocity_branch)
                + K
            )

        if element["direction"] == "bend":

            # dampening:
            # section 11.2 a)
            speed_of_sound = 340

            f_G = 0.586 * speed_of_sound / diameter_branch

            def find_category(value, intervals):
                """
                Finds the category where the value fits in
                """
                index = np.searchsorted(intervals, value, side="right")
                return index

            # section 11.2 b)
            category = find_category(f_G, f_o)

            # section 11.2 c)
            def custom_shift(arr, shift_val):
                # Create an array of zeros with the same shape as arr
                shifted = np.zeros_like(arr)

                # If shifting to the right
                if shift_val > 0:
                    shifted[shift_val:] = arr[:-shift_val]
                # If shifting to the left
                elif shift_val < 0:
                    shifted[:shift_val] = arr[-shift_val:]
                # If no shift, simply return the original array
                else:
                    return arr

                return shifted

            dampening_levels = np.array([0, 1, 2, 3, 3, 3, 3, 3, 3])
            dampening = custom_shift(dampening_levels, category - 2)[:-1]
        elif element["direction"] == "straight":
            dampening = np.zeros(8)
        else:
            raise ValueError(
                f"Direction {element['direction']} should be bend / straight"
            )

        dampening = dampening + np.abs(10 * np.log10(A_branch / (A_branch + A_neighbour_branch)))

        return dampening, flow_noise

    # ---------------------- orchestration -----------------------
    def step(self, element: dict) -> None:
        """Process a single element and append to history."""

        try:
            if element["name"] in self.ignore_components:
                print(f"Element {element['name']} is ignored.")
            else:
                func = self._dispatch()[element["name"]]
                func(element)
        except KeyError as exc:
            raise KeyError(f"Unknown element name: {element.get('name')!r}") from exc

        behind = {
            "zeta": self.zeta,
            "volume_flow_ratio": self.volume_flow_ratio,
            "width": self.width,
            "height": self.height,
            "volume_flow": self.volume_flow,
            "dampening": self.dampening,
            "flow_noise": self.flow_noise,
            "name": element.get("name"),
        }
        self.history.append({"element": element, "behind": behind})

    def run(self, data: dict):
        """
        Run the whole system described by a loaded YAML dict.
        Returns (total_zeta, zeta_history, volume_flow_ratio_history).
        """
        # init state from data
        sys_w = data["system"]["width"]
        sys_h = data["system"]["height"]
        self.reset(width=sys_w, height=sys_h)

        # iterate in defined order (Python 3.7+ preserves dict order)
        for _, element in data["system"]["elements"].items():
            self.step(element)

        # zeta_hist = [h["after"]["zeta"] for h in self.history]
        # vfr_hist = [h["after"]["volume_flow_ratio"] for h in self.history]
        return self.zeta

    @classmethod
    def from_yaml(
        cls,
        data: str,
        volume_flow: float = 0,
        include_acoustics=False,
        ignore_components=[],
    ):
        """
        Convenience: load a YAML via pyomo2h5.load_yaml, construct the calculator,
        run it, and return (calc, data, total_zeta, zeta_history, vfr_history).
        """
        if isinstance(data,str):
            from pyomo2h5 import load_yaml  # local import to avoid hard dependency
            data = load_yaml(data)
        
        calc = cls(
            width=data["system"]["width"],
            height=data["system"]["height"],
            volume_flow=volume_flow,
            ignore_components=ignore_components,
            include_acoustics=include_acoustics,
        )
        calc.run(data)
        return calc

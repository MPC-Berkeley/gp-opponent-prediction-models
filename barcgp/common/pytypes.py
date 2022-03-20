from dataclasses import dataclass, field
import array
import numpy as np
import copy


# DEFAULT_VEHICLE_TYPE = 'barc'


@dataclass
class PythonMsg:
    '''
    Base class for python messages. Intention is that fields cannot be changed accidentally,
    e.g. if you try "state.xk = 10", it will throw an error if no field "xk" exists.
    This helps avoid typos. If you really need to add a field use "object.__setattr__(state,'xk',10)"
    Dataclasses autogenerate a constructor, e.g. a dataclass fields x,y,z can be instantiated
    "pos = Position(x = 1, y = 2, z = 3)" without you having to write the __init__() function, just add the decorator
    Together it is hoped that these provide useful tools and safety measures for storing and moving data around by name,
    rather than by magic indices in an array, e.g. q = [9, 4.5, 8829] vs. q.x = 10, q.y = 9, q.z = 16
    '''

    def __setattr__(self, key, value):
        '''
        Overloads default attribute-setting functionality to avoid creating new fields that don't already exist
        This exists to avoid hard-to-debug errors from accidentally adding new fields instead of modifying existing ones

        To avoid this, use:
        object.__setattr__(instance, key, value)
        ONLY when absolutely necessary.
        '''
        if not hasattr(self, key):
            raise TypeError('Cannot add new field "%s" to frozen class %s' % (key, self))
        else:
            object.__setattr__(self, key, value)

    def print(self, depth=0, name=None):
        '''
        default __str__ method is not easy to read, especially for nested classes.
        This is easier to read but much longer

        Will not work with "from_str" method.
        '''
        print_str = ''
        for j in range(depth): print_str += '  '
        if name:
            print_str += name + ' (' + type(self).__name__ + '):\n'
        else:
            print_str += type(self).__name__ + ':\n'
        for key in vars(self):
            val = self.__getattribute__(key)
            if isinstance(val, PythonMsg):
                print_str += val.print(depth=depth + 1, name=key)
            else:
                for j in range(depth + 1): print_str += '  '
                print_str += str(key) + '=' + str(val)
                print_str += '\n'

        if depth == 0:
            print(print_str)
        else:
            return print_str

    def from_str(self, string_rep):
        '''
        inverts dataclass.__str__() method generated for this class so you can unpack objects sent via text (e.g. through multiprocessing.Queue)
        '''
        val_str_index = 0
        for key in vars(self):
            val_str_index = string_rep.find(key + '=', val_str_index) + len(key) + 1  # add 1 for the '=' sign
            value_substr = string_rep[val_str_index: string_rep.find(',',
                                                                     val_str_index)]  # (thomasfork) - this should work as long as there are no string entries with commas

            if '\'' in value_substr:  # strings are put in quotes
                self.__setattr__(key, value_substr[1:-1])
            if 'None' in value_substr:
                self.__setattr__(key, None)
            else:
                self.__setattr__(key, float(value_substr))

    def copy(self):
        return copy.deepcopy(self)


@dataclass
class NodeParamTemplate:
    '''
    Base class for node parameter templates
    used by autodeclaring and loading parameters as implemented in mpclab_base_nodes.py
    This class provides functionality for turning a generated template into a default configuration file (.yaml format)
    This itself is not a template. To create a template create a class (preferably in the source code for the node itself)
    and in __init__() add attributes to the class, e.g. self.dt = 0.1, self.state = VehicleCoords(), etc...
    These must be added in __init__(), not outside of it, or the variables wont show up in vars(instace_of_template).
    '''

    def spew_yaml(self):
        def append_param(yaml_str, key, val, indent_depth):
            for j in range(indent_depth):
                yaml_str += '  '

            yaml_str += key
            yaml_str += ': '
            if isinstance(val, str):
                yaml_str += "'" + val + "'"
            elif isinstance(val, np.ndarray):
                yaml_str += val.tolist().__str__()
            elif isinstance(val, (bool, int, float, str)):
                yaml_str += val.__str__()
            elif isinstance(val, (list, tuple, array.array)):
                yaml_str += val.__str__()
            elif val is None:
                yaml_str += ''
            else:
                yaml_str += '0'

            yaml_str += '\n'
            return yaml_str

        def unpack_pythonmsg(yaml_str, msg, prefix, depth=2):
            yaml_str = append_param(yaml_str, prefix, None, depth)
            for key in vars(msg):
                val = msg.__getattribute__(key)
                if isinstance(val, PythonMsg):
                    yaml_str = unpack_pythonmsg(yaml_str, val, key, depth + 1)
                else:
                    yaml_str = append_param(yaml_str, key, val, depth + 1)

            return yaml_str

        yaml_str = 'namespace:\n'
        yaml_str += '  ros__parameters:\n'

        parameters = []
        for key in vars(self):
            val = self.__getattribute__(key)
            if isinstance(val, PythonMsg):
                yaml_str = unpack_pythonmsg(yaml_str, val, key)
            else:
                yaml_str = append_param(yaml_str, key, val, 2)

        return yaml_str


@dataclass
class Position(PythonMsg):
    x: float = field(default=0)
    y: float = field(default=0)
    z: float = field(default=0)


@dataclass
class VehicleActuation(PythonMsg):
    t: float = field(default=0)

    u_a: float = field(default=0)
    u_steer: float = field(default=0)

    def __str__(self):
        return 't:{self.t}, u_a:{self.u_a}, u_steer:{self.u_steer}'.format(self=self)


@dataclass
class TrackLookahead(PythonMsg):
    '''
    Local track information ahead of the vehicle (curvature)
    '''
    t: float = field(default=None)  # time in seconds

    l: float = field(default=None)  # length of lookahead in meters
    dl: float = field(default=None)  # discretization step-size of the lookahead
    n: int = field(default=None)  # length of lookahead in array entries

    # TODO Add field for segmented lookahead?
    curvature: array.array = field(default=None)  # the curvature lookahead

    def __post_init__(self):
        if self.l is None: self.l = 1.5
        if self.dl is None: self.dl = 0.5
        self.n = int(round(self.l / self.dl))
        dummyList = self.n * [1.0]
        self.curvature = array.array("d")
        self.curvature.extend(dummyList)

    # TODO: should this be updated from within the class? e.g. call the update every time-step? Probably not


@dataclass
class BodyLinearVelocity(PythonMsg):
    v_long: float = field(default=0)
    v_tran: float = field(default=0)
    v_n: float = field(default=0)


@dataclass
class BodyAngularVelocity(PythonMsg):
    w_phi: float = field(default=0)
    w_theta: float = field(default=0)
    w_psi: float = field(default=0)


@dataclass
class BodyLinearAcceleration(PythonMsg):
    a_long: float = field(default=0)
    a_tran: float = field(default=0)
    a_n: float = field(default=0)


@dataclass
class BodyAngularAcceleration(PythonMsg):
    a_phi: float = field(default=0)
    a_theta: float = field(default=0)
    a_psi: float = field(default=0)


@dataclass
class OrientationEuler(PythonMsg):
    phi: float = field(default=0)
    theta: float = field(default=0)
    psi: float = field(default=0)


@dataclass
class ParametricPose(PythonMsg):
    s: float = field(default=0)
    x_tran: float = field(default=0)
    n: float = field(default=0)
    e_psi: float = field(default=0)


@dataclass
class ParametricVelocity(PythonMsg):
    ds: float = field(default=0)
    dx_tran: float = field(default=0)
    dn: float = field(default=0)
    de_psi: float = field(default=0)


@dataclass
class VehicleState(PythonMsg):
    '''
    Complete vehicle state (local, global, and input)
    '''
    t: float = field(default=None)  # time in seconds

    x: Position = field(default=None)  # global position

    v: BodyLinearVelocity = field(default=None)  # body linear velocity
    w: BodyAngularVelocity = field(default=None)  # body angular velocity
    a: BodyLinearAcceleration = field(default=None)  # body linear acceleration
    aa: BodyAngularAcceleration = field(default=None)  # body angular acceleration

    e: OrientationEuler = field(default=None)  # global orientation (phi, theta, psi)

    p: ParametricPose = field(default=None)  # parametric position (s,y, ths)
    pt: ParametricVelocity = field(default=None)  # parametric velocity (ds, dy, dths)

    u: VehicleActuation = field(default=None)

    lookahead: TrackLookahead = field(default=None)  # TODO Find a good field name :(

    v_x: float = field(default=0)
    v_y: float = field(default=0)

    lap_num: int = field(default=None)

    def __post_init__(self):
        if self.x is None: self.x = Position()
        if self.u is None: self.u = VehicleActuation()
        if self.lookahead is None: self.lookahead = TrackLookahead()
        if self.v is None: self.v = BodyLinearVelocity()
        if self.w is None: self.w = BodyAngularVelocity()
        if self.a is None: self.a = BodyLinearAcceleration()
        if self.aa is None: self.aa = BodyAngularAcceleration()
        if self.e is None: self.e = OrientationEuler()
        if self.p is None: self.p = ParametricPose()
        if self.pt is None: self.pt = ParametricVelocity()
        return

    def update_global_velocity_from_body(self):
        self.v_x = self.v.v_long * np.cos(self.e.psi) - self.v.v_tran * np.sin(self.e.psi)
        self.v_y = self.v.v_long * np.sin(self.e.psi) + self.v.v_tran * np.cos(self.e.psi)
        # self.a_x =  self.a.a_long * np.cos(self.psi) - self.a_tran * np.sin(self.psi)
        # self.a_y =  self.a.a_long * np.sin(self.psi) + self.a_tran * np.cos(self.psi)

    def copy_control(self, destination):
        '''
        copies control state form self to destination
        '''
        destination.t = self.t
        destination.u_a = self.u.u_a
        destination.u_steer = self.u.u_steer
        return


@dataclass
class VehiclePrediction(PythonMsg):
    '''
    Complete vehicle coordinates (local, global, and input)
    '''
    t: float = field(default=None)  # time in seconds

    x: array.array = field(default=None)  # global x coordinate in meters
    y: array.array = field(default=None)  # global y coordinate in meters

    v_x: array.array = field(default=None)  # global x velocity in m/s
    v_y: array.array = field(default=None)  # global y velocity in m/s

    a_x: array.array = field(default=None)  # global x acceleration in m/s^2
    a_y: array.array = field(default=None)  # global y acceleration in m/s^2

    psi: array.array = field(default=None)  # global vehicle heading angle
    psidot: array.array = field(default=None)  # global and local angular velocity of car

    v_long: array.array = field(default=None)  # longitudinal velocity (in the direction of psi)
    v_tran: array.array = field(default=None)  # transverse velocity   (orthogonal to the direction of psi)

    a_long: array.array = field(default=None)  # longitudinal velocity (in the direction of psi)
    a_tran: array.array = field(default=None)  # transverse velocity   (orthogonal to the direction of psi)

    e_psi: array.array = field(default=None)  # heading error between car and track
    s: array.array = field(default=None)  # path length along center of track to projected position of car
    x_tran: array.array = field(default=None)  # deviation from centerline (transverse position)

    u_a: array.array = field(default=None)  # acceleration output
    u_steer: array.array = field(default=None)  # steering angle output

    lap_num: int = field(default=None)

    sey_cov: np.array = field(default=None)

    xy_cov: np.array = field(default=None)  # covariance matrix in local heading frame

    def update_body_velocity_from_global(self):
        self.v_long = (np.multiply(self.v_x, np.cos(self.psi)) + np.multiply(self.v_y, np.sin(self.psi))).tolist()
        self.v_tran = (-np.multiply(self.v_x, np.sin(self.psi)) + np.multiply(self.v_y, np.cos(self.psi))).tolist()
        self.a_long = (np.multiply(self.a_x, np.cos(self.psi)) + np.multiply(self.a_y, np.sin(self.psi))).tolist()
        self.a_tran = (-np.multiply(self.a_y, np.sin(self.psi)) + np.multiply(self.a_y, np.cos(self.psi))).tolist()

    def update_global_velocity_from_body(self):
        self.v_x = (np.multiply(self.v_long, np.cos(self.psi)) - np.multiply(self.v_tran, np.sin(self.psi))).tolist()
        self.v_y = (np.multiply(self.v_long, np.sin(self.psi)) + np.multiply(self.v_tran, np.cos(self.psi))).tolist()
        self.a_x = (np.multiply(self.a_long, np.cos(self.psi)) - np.multiply(self.a_tran, np.sin(self.psi))).tolist()
        self.a_y = (np.multiply(self.a_long, np.sin(self.psi)) + np.multiply(self.a_tran, np.cos(self.psi))).tolist()

    def track_cov_to_local(self, track, N : int, cov_factor : float):
        """
        Converts s, x_tran uncertainty to uncertainty in car heading direction, NOT global frame!
        """
        self.xy_cov = np.zeros((N, 2, 2))
        if self.sey_cov is not None:
            sey_unflat = np.array(self.sey_cov).reshape(N, 4)
            for i in range(1, N):
                sey_cov = sey_unflat[i].reshape(2, 2)
                c = track.get_curvature(self.s[i])
                if not c == 0:
                    radius = 1/c
                    if radius > 0:  # left turn
                        sey_cov[0,0] = sey_cov[0,0]*(radius - self.x_tran[i])/radius
                    else:  #  right turn, is this correct?
                        sey_cov[0, 0] = sey_cov[0, 0] * (radius - self.x_tran[i]) / radius
                angle = self.e_psi[i]
                # TODO Make this covariance
                self.xy_cov[i] = np.array(
                    [[np.fabs(np.cos(angle)**2 * sey_cov[0, 0] + np.sin(angle)**2 * sey_cov[1, 1]), 0],[0,np.fabs(np.sin(angle)**2 * sey_cov[0, 0] + np.cos(angle)**2 * sey_cov[1, 1])]])
                self.xy_cov[i] *= cov_factor
                

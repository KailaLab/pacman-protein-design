import sys
import importlib
import inspect
import os

plugin_directory = os.path.join(os.path.dirname(__file__), 'plugins')


class PluginImportError(ImportError):
    """ raise when a plugin could not be loaded"""

    def __init__(self, message, *args):
        self.message = message

        super(PluginImportError, self).__init__(message, *args)


class PluginReturnError(ValueError):
    """ raise when a plugin could not be loaded"""

    def __init__(self, message, *args):
        self.message = message

        super(PluginReturnError, self).__init__(message, *args)


class PluginArgError(KeyError):

    def __init__(self, message, *args):
        self.message = message

        super(PluginArgError, self).__init__(message, *args)


class Plugin(object):
    """ A simple wrapper for some static methods."""

    def __init__(self, plugin_name, plugin_pars):

        self.name = plugin_name
        self.plugin_directory = os.path.join(os.path.dirname(__file__), 'plugins')
        self.func = self.load_plugin(plugin_name, self.plugin_directory)
        self.plugin_pars = plugin_pars

    @staticmethod
    def load_plugin(plugin_name, direc):
        """
        Imports the run function of a plugin
        :param plugin_name: name of the plugin to be loaded
        :type plugin_name: str
        :param direc: directory in which plugin subdirectories are
        :type direc: str
        :return: run function defined as run.py in the plugin subdirectory
        :rtype: function
        """
        filename = 'run.py'

        plugin_dir = os.path.join(direc, plugin_name)

        if not os.path.isdir(plugin_dir):
            mes = 'Folder for plugin ({}) could not been found!'
            mes += 'Make sure it is in the plugin dir!'
            mes = mes.format(plugin_name)
            raise PluginImportError(mes)

        if filename not in os.listdir(plugin_dir):
            mes = 'File `run.py` was not found in the plugin directory'
            raise PluginImportError(mes)

        plugin = importlib.import_module("{}.{}.{}".format('plugins', plugin_name, "run"))

        if not inspect.isfunction(plugin.run):
            mes = 'Plugin `{}` does not provide a valid run function!'
            raise PluginImportError(mes)

        return plugin.run

    @staticmethod
    def get_plugin_pars(plugin_name):
        """
        Read a dictionary of additional paramters to be parsed by Parser.py
        :param plugin_name: name of the plugin for which parameters are to be read
        :type plugin_name: str
        :return: The additional plugin paramteres to be parsed
        :rtype: dict
        """
        plugin_dir = os.path.join(plugin_directory, plugin_name)
        filename = "parameters.py"

        if filename not in os.listdir(plugin_dir):
            mes = 'File `run.py` was not found in the plugin directory'
            raise PluginImportError(mes)

        plugin = importlib.import_module("{}.{}.{}".format('plugins', plugin_name, "parameters"))

        try:
            parameters = plugin.parameters
        except NameError:
            parameters = {}

        return parameters

    def exec_plugin(self, snap):
        """
        Wrapper for the plugin function
        :param snap: An instance of the Snap class defined in Snap.py for making pacman information available to plugin
        :type snap: object
        :return: Property evaluated by the plugin (usually an energy in J/mol and a common unit for the specific energy)
        :rtype: tuple
        """
        try:

            ret_val = self.func(snap, self.plugin_pars)

        except TypeError:

            try:
                ret_val = self.func(snap)
            except Exception as e:
                raise type(e)(str(e) + 'Error in plugin {}!'.format(self.name)).with_traceback(sys.exc_info()[2])

        except Exception as e:
            raise type(e)(str(e) + 'Error in plugin {}!'.format(self.name)).with_traceback(sys.exc_info()[2])

        if isinstance(type(ret_val), type((1, 2))):
            raise PluginReturnError('Plugin {}: type returned is {}! Please return tuple!'.format(self.name,
                                                                                                  type(ret_val)))

        if len(ret_val) != 2:
            raise PluginReturnError('Plugins must return a tuple of length 2! Plugin {} returned {}!'.format(self.name,
                                                                                                             ret_val))
        try:
            plugin_return = tuple((float(i) * float(self.plugin_pars['w']) for i in ret_val))
        except ValueError:
            mes = 'Plugin {}: plugins may only return float values in the tuple! Plugin returned {}!'.format(self.name,
                                                                                                             ret_val)
            raise PluginReturnError(mes)

        return plugin_return


# only for testing
if __name__ == '__main__':

    p = Plugin('random', None)

    p.exec_plugin('test')

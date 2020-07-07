from PyQt5 import QtWidgets, QtCore
from Enthalpy import Ui_Dialog
from GetInput_Mol_TS import ExtendConvertToGraph
import tensorflow as tf
import numpy as np

class mywindow(QtWidgets.QWidget, Ui_Dialog):
    def  __init__ (self):
        super(mywindow, self).__init__()
        self.setupUi(self)
        self.pushButton_ok.clicked.connect(self.search_enthalpy)
        self.pushButton_ok.clicked.connect(self.give_type)
        self.pushButton_ok.clicked.connect(self.predict_enthalpy)

    def search_enthalpy(self):
        #选取文件
        f = open('./Arkane_output.py', 'r')
        lines = f.readlines()
        key_word = 'label = \'' + self.lineEdit_1.text() + '\''
        # print(key_word)
        iter = 0
        Enthalpy_str = 'Sorry, this species is not in database'
        for line in lines:
            iter += 1
            if key_word in line:
                Enthalpy_line = lines[iter]
                idx1 = Enthalpy_line.index('(')
                idx2 = Enthalpy_line.index(',')
                Enthalpy_tran = float(Enthalpy_line[idx1+1:idx2]) * 0.2389
                Enthalpy_str = str(np.round(Enthalpy_tran, decimals=3)) + ' kcal/mol'
        self.lineEdit_2.setText(Enthalpy_str)

    def give_type(self):
        line = self.lineEdit_1.text()
        if '_' in line:  # judge TS or Mol
            underline = line.index('_')
            if '+' in line[0:underline]:  # H_abstraction or R_addition_MultipleBond
                product = line[underline+1:]
                if '+' in product:  # Hydrogen_abstraction
                    self.label_1.setText('TS_H-abstraction')
                else:  # R_addition_MultipleBond
                    self.label_1.setText('TS_R-addition_multibond')
            else:                       # H_migration
                self.label_1.setText('TS_Intra_H-migration')
        else:
            if '[' in line:
                self.label_1.setText('Radical')
            else:
                self.label_1.setText('Molecule')

    def predict_enthalpy(self):
        features, adj = ExtendConvertToGraph([self.lineEdit_1.text()])
        features = features.reshape((1, 12, 36))
        adj = adj.reshape((1, 12, 12))

        with tf.Session() as sess:
            new_saver = tf.train.import_meta_graph('./Save_model/GCN+a+g_3_Adam_256_0.003.ckpt.meta')
            new_saver.restore(sess, tf.train.latest_checkpoint('./Save_model'))
            graph = tf.get_default_graph()
            input_A = graph.get_operation_by_name('input_A').outputs[0]
            input_X = graph.get_operation_by_name('input_X').outputs[0]
            feed_dict = {input_A: adj, input_X: features}
            prop_pred = tf.get_collection('pred_prop')[0]
            prediction = sess.run(prop_pred, feed_dict=feed_dict)
            prediction_actual = prediction * 239.0273 - 105.3774
            pred_str = str(np.round(prediction_actual[0][0], decimals=3)) + ' kcal/mol'
        self.lineEdit_3.setText(pred_str)

if __name__=="__main__":
    import sys
    QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)
    app=QtWidgets.QApplication(sys.argv)
    ui = mywindow()
    ui.show()
    sys.exit(app.exec_())
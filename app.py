import os
from flask import Flask, render_template, request, jsonify, send_from_directory, url_for
from dotenv import load_dotenv
import logging
import ee
import time
from methane_detection import process_emit_data
import traceback

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def ensure_directory_exists(directory):
    """Create directory if it doesn't exist"""
    if not os.path.exists(directory):
        os.makedirs(directory)
        logger.info(f"Created directory: {directory}")

def init_earth_engine():
    """Initialize Earth Engine"""
    try:
        ee.Initialize(project='ee-rohithreddy66666')
        logger.info("Earth Engine initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize Earth Engine. Error: {e}")
        logger.error("Please ensure you have authenticated with Earth Engine.")

def create_app():
    # Load environment variables
    load_dotenv()
    
    # Initialize Earth Engine
    init_earth_engine()
    
    # Initialize Flask app
    app = Flask(__name__,
        static_folder='templates/assets',
        static_url_path='/assets')
    
    # Configure app directories
    app.config['DATA_FOLDER'] = os.path.join(app.root_path, 'data')
    app.config['UPLOAD_FOLDER'] = os.path.join(app.root_path, 'uploads')
    app.config['RESULTS_FOLDER'] = os.path.join(app.root_path, 'templates/assets/results')
    
    # Ensure all required directories exist
    ensure_directory_exists(app.config['DATA_FOLDER'])
    ensure_directory_exists(app.config['UPLOAD_FOLDER'])
    ensure_directory_exists(app.config['RESULTS_FOLDER'])
    
    # Main page routes
    @app.route('/')
    def index():
        return render_template('index.html')

    @app.route('/methane')
    def methane():
        return render_template('methane.html')

    @app.route('/time')
    def time_series():
        return render_template('time.html')

    # Methane Detection Routes
    @app.route('/process', methods=['POST'])
    def process():
        try:
            # Get EMIT ID from request
            emit_id = request.json.get('emit_id')
            if not emit_id:
                return jsonify({
                    'success': False,
                    'error': 'EMIT ID is required'
                }), 400

            logger.info(f"Starting processing for EMIT ID: {emit_id}")
            start_time = time.time()
            
            try:
                results = process_emit_data(emit_id, output_dir='templates/assets/results')
                
                # Update visualization path
                if results.get('visualization_path'):
                    results['visualization_path'] = url_for('static', 
                        filename=f'results/{os.path.basename(results["visualization_path"])}')
                    logger.info(f"Image path set to: {results['visualization_path']}")
                
            except Exception as e:
                logger.error(f"Error in process_emit_data: {str(e)}")
                raise
            
            processing_time = time.time() - start_time
            logger.info(f"Processing completed in {processing_time:.2f} seconds")
            
            return jsonify({
                'success': True,
                'results': results,
                'processing_time': processing_time,
                'message': 'Processing completed successfully'
            })

        except Exception as e:
            logger.error("Error occurred during processing:")
            logger.error(traceback.format_exc())
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500

    @app.route('/results/<path:filename>')
    def get_result(filename):
        """Serve result files"""
        try:
            return send_from_directory('templates/assets/results', filename)
        except Exception as e:
            logger.error(f"Error serving file {filename}: {str(e)}")
            return jsonify({'error': f'File not found: {str(e)}'}), 404

    # Utility Routes
    @app.route('/status')
    def status():
        """Health check endpoint"""
        return jsonify({
            'status': 'running',
            'message': 'API is operational'
        })

    @app.route('/debug/results')
    def debug_results():
        """Debug endpoint to check results directory"""
        results_dir = os.path.join('templates', 'assets', 'results')
        try:
            files = os.listdir(results_dir)
            return jsonify({
                'results_dir': results_dir,
                'files': files,
                'static_folder': app.static_folder,
                'static_url_path': app.static_url_path,
                'exists': os.path.exists(results_dir),
                'is_dir': os.path.isdir(results_dir),
                'permissions': oct(os.stat(results_dir).st_mode)[-3:],
                'absolute_path': os.path.abspath(results_dir)
            })
        except Exception as e:
            return jsonify({
                'error': str(e),
                'results_dir': results_dir,
                'static_folder': app.static_folder,
                'static_url_path': app.static_url_path
            })

    @app.route('/debug/directories')
    def debug_directories():
        """Debug endpoint to check all relevant directories"""
        directories = {
            'static': app.static_folder,
            'templates': app.template_folder,
            'uploads': 'uploads',
            'results': 'templates/assets/results'
        }
        
        status = {}
        for name, path in directories.items():
            status[name] = {
                'path': path,
                'exists': os.path.exists(path),
                'is_directory': os.path.isdir(path) if os.path.exists(path) else False,
                'absolute_path': os.path.abspath(path),
                'contents': os.listdir(path) if os.path.exists(path) and os.path.isdir(path) else None
            }
        
        return jsonify(status)

    # Error Handlers
    @app.errorhandler(404)
    def not_found_error(error):
        return jsonify({'error': 'Not found'}), 404

    @app.errorhandler(500)
    def internal_error(error):
        return jsonify({'error': 'Internal server error'}), 500

    # Register time series API blueprint
    from api import api_bp
    app.register_blueprint(api_bp, url_prefix='/api')
    
    return app

if __name__ == '__main__':
    logger.info("Starting Combined Methane Detection and Analysis Server...")
    app = create_app()
    logger.info("Server is ready!")
    app.run(debug=True, port=5000)